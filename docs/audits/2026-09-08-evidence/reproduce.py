"""Reproduce the 2026-09-08 audit findings using synthetic data only."""

import json
import logging
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import src.data_ingestion as ingestion
from src.analytics import (
    calculate_daily_spreads,
    calculate_negative_price_hours,
    calculate_spread_percentiles,
    calculate_two_stage_da_id_dispatch,
)
from src.data_ingestion import (
    build_zone_query_window,
    clean_prices,
    parse_intraday_csv,
    persist_intraday_frame,
    read_intraday_cache,
)
from src.dispatch import solve_joint_capacity_batch
from src.pages.market_overview import render as render_market_overview
from src.pages.simulation_cockpit import _reserve_coopt_total
from src.project_case import CurrencyBasis, CurrencyBasisMode, emit_reserve_coopt, grid
from src.simulation import simulate_da_id_replay, simulate_da_milp_replay, simulate_replay_batch
from src.ui_theme import cockpit_chart_template

# 1. Native market mixed-cadence DE window vs physically identical regular representation.
i1 = pd.date_range("2025-09-30", periods=24, freq="h", tz="Europe/Berlin")
i2 = pd.date_range("2025-10-01", periods=96, freq="15min", tz="Europe/Berlin")
native = pd.DataFrame(
    {"price_eur_mwh": [20.0] * 12 + [100.0] * 12 + [60.0] * 96},
    index=i1.append(i2).tz_convert("UTC"),
)
regular_index = pd.date_range(i1[0], periods=192, freq="15min").tz_convert("UTC")
regular = native.reindex(regular_index, method="ffill")
for name, frame in [("native", native), ("physical_15m_equivalent", regular)]:
    batch = simulate_replay_batch(
        frame, tz="Europe/Berlin", duration_hours=4.0, efficiency=1.0, soc_init_frac=0.0
    )
    print(
        "mixed_continuous",
        name,
        json.dumps(
            batch[["date", "total_revenue_eur", "physical_throughput_mwh", "n_intervals"]].to_dict(
                "records"
            ),
            default=str,
        ),
        batch.attrs,
    )

# 2. Physical boundary at 00 CET occurs inside FI civil day.
start, end = build_zone_query_window("FI", "2025-10-01", "2025-10-01")
transition = pd.Timestamp("2025-10-01", tz="Europe/Berlin").tz_convert("UTC")
idx = pd.date_range(start, transition, freq="h", inclusive="left").append(
    pd.date_range(transition, end, freq="15min", inclusive="left")
)
fi = clean_prices(
    pd.DataFrame({"price_eur_mwh": 50.0}, index=idx),
    zone="FI",
    expected_start=start,
    expected_end=end,
)
print(
    "fi_clean",
    len(fi),
    "filled",
    int(fi.filled.sum()),
    "actual_hours",
    (end - start).total_seconds() / 3600,
)
print(
    "fi_replay",
    simulate_da_milp_replay(fi, simulation_date=date(2025, 10, 1), tz="Europe/Helsinki")["summary"],
)
print(
    "fi_reserve",
    solve_joint_capacity_batch(fi, 5.0, tz="Europe/Helsinki", efficiency=1.0)[
        ["joint_capacity_revenue"]
    ].to_dict("records"),
    "expected",
    24 * 5 * 0.95,
)

# 3. Missing IDA interval accepted in Revenue but rejected in cockpit.
idx = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
da = pd.DataFrame({"price_eur_mwh": [20.0] * 12 + [100.0] * 12}, index=idx)
ida = da.rename(columns={"price_eur_mwh": "intraday_price_eur_mwh"}).drop(idx[8])
result = calculate_two_stage_da_id_dispatch(da, ida, tz="UTC", efficiency=1.0)
print("sparse_ida_analytics", result.to_dict("records"), result.attrs)
print(
    "sparse_ida_replay",
    simulate_da_id_replay(da, ida, simulation_date=date(2026, 1, 1), tz="UTC")["summary"],
)
# 4. DA/IDA differently-sized delivery intervals inner-joined on starts.
daidx = pd.date_range("2025-09-01", periods=24, freq="h", tz="UTC")
idaidx = pd.date_range("2025-09-01", periods=96, freq="15min", tz="UTC")
da = pd.DataFrame({"price_eur_mwh": 50.0}, index=daidx)
ida = pd.DataFrame({"intraday_price_eur_mwh": np.tile([20.0, 100.0, 20.0, 20.0], 24)}, index=idaidx)
for name, frame in [
    ("native_hourly_DA", da),
    ("explicit_15m_DA", da.reindex(idaidx, method="ffill")),
]:
    result = simulate_da_id_replay(
        frame, ida, simulation_date=date(2025, 9, 1), tz="UTC", efficiency=1.0, soc_init_frac=0.0
    )
    print(
        "mixed_markets",
        name,
        "n",
        len(result["timeseries"]),
        "rev",
        result["summary"]["total_revenue_eur"],
        "success",
        result["summary"]["success"],
    )
# 5. Infinity survives manual import and durable cache, then violates typed solver failure contract.


idx = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
csv = "timestamp,ida_price_eur_mwh\n" + "\n".join(
    f"{ts.isoformat()},{'inf' if i == 12 else '50'}" for i, ts in enumerate(idx)
)
parsed = parse_intraday_csv(csv, default_zone="DE_LU")
with (
    tempfile.TemporaryDirectory(prefix="euro-bess-core-audit-") as tmp,
    patch.object(ingestion, "DB_PATH", Path(tmp) / "cache.db"),
):
    summary = persist_intraday_frame(parsed)
    cached = read_intraday_cache("DE_LU", idx[0], idx[-1] + pd.Timedelta(hours=1))
    print(
        "inf_import",
        "persisted",
        summary,
        "infinity_in_cache",
        int(np.isinf(cached.intraday_price_eur_mwh).sum()),
    )
    da = pd.DataFrame({"price_eur_mwh": 50.0}, index=idx)
    try:
        r = simulate_da_id_replay(da, cached, simulation_date=date(2026, 1, 1), tz="UTC")
        print("inf_replay", r["summary"])
    except Exception as exc:
        print("inf_replay_throws", type(exc).__name__, str(exc))

# 6. The plotted 30-Day MA is a row count, so quarter-hour data spans 7.5 days.

for logger_name in logging.Logger.manager.loggerDict:
    if logger_name.startswith("streamlit"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

idx = pd.date_range("2026-01-01", periods=40 * 96, freq="15min", tz="UTC", name="timestamp")
frame = pd.DataFrame(
    {"price_eur_mwh": np.r_[np.full(30 * 96, 10.0), np.full(10 * 96, 100.0)]}, index=idx
)
daily = calculate_daily_spreads(frame, tz="UTC")
figures = {}
render_market_overview(
    "DE_LU",
    frame,
    daily,
    calculate_spread_percentiles(daily),
    calculate_negative_price_hours(frame),
    1,
    "UTC",
    cockpit_chart_template(),
    figures,
)
print(
    "plotted_ma",
    figures["price_ts"].data[1].name,
    float(figures["price_ts"].data[1].y[-1]),
    "true_30d",
    float(frame.price_eur_mwh.rolling("30D").mean().iloc[-1]),
)

# 7. Nominal German reserve-block settlement versus legacy physical-hour consumers.

for day in [date(2025, 3, 30), date(2025, 10, 26)]:
    da = pd.DataFrame(
        {"price_eur_mwh": 0.0}, index=pd.DatetimeIndex(grid.expected_da_timestamps("DE_LU", day))
    )
    reserve = pd.Series(
        20.0, index=pd.DatetimeIndex([bid for bid, _ in grid.reserve_blocks("DE_LU", day)])
    )
    anc = pd.DataFrame(
        {
            "product_type": "FCR",
            "direction": "symmetric",
            "zone": "DE_LU",
            "capacity_price_eur_mw": reserve,
        },
        index=reserve.index,
    )
    pc = emit_reserve_coopt(
        da,
        reserve,
        zone="DE_LU",
        first_delivery_date=day,
        last_delivery_date=day,
        power_mw=1.0,
        duration_hours=1.0,
        efficiency=0.88,
        currency_basis=CurrencyBasis(CurrencyBasisMode.SOURCE_EUR_TREATED_AS_BASE_YEAR_REAL, 2025),
        reserve_product="FCR",
        reserve_source="audit_synthetic",
    )
    cockpit = _reserve_coopt_total(
        da,
        "FCR",
        anc,
        valid_dates={day},
        tz="Europe/Berlin",
        power_mw=1.0,
        duration_hours=1,
        efficiency=0.88,
    )
    print(
        "dst_reserve",
        day,
        "project_case",
        pc.daily_realised_cash_series[0][1],
        "cockpit",
        cockpit[0],
    )
