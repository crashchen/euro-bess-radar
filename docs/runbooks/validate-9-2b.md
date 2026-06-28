# Runbook: validate the Phase 9.2b cockpit panel end-to-end

The live IDA auction API returns no data and capacity prices only exist for a
few zones, so the Phase 9.2b forecast-driven reserve panel cannot be exercised
from live data alone. This runbook seeds a **synthetic** DE_LU dataset and walks
through validating the panel, the export, and the Data Trust coverage matrix.

> The seeded data is synthetic (IDA1 is labelled `Synthetic demo` in Data
> Trust). It is for UI/feature validation only, **not** a market estimate.

## 1. Seed the demo data

```bash
source .venv/bin/activate
python scripts/seed_demo_9_2b.py            # ~30 days ending today; --days N to change
```

This writes synthetic DA + IDA1 into the local SQLite cache and emits
`samples/de_lu_reserve_capacity_sample.csv` (DE_FCR template). Note the local
date window it prints.

Safety guard: the script refuses to overwrite existing DE_LU cache tables unless
they were created by a previous `Synthetic demo` run. If you intentionally want
to replace local DE_LU cache data for this validation, back it up first and run
with `--force`.

## 2. Launch and load

```bash
streamlit run app.py
```

1. Select zone **DE_LU** and a date range inside the printed window.
2. Sidebar → **Ancillary upload** → template **DE_FCR** → upload
   `samples/de_lu_reserve_capacity_sample.csv`.

## 3. Verify the cockpit panel

Open **Simulation Cockpit → "Forecast-driven IDA policy"**, choose the reserve
product, and click **Run**. Check:

- [ ] Strategy comparison shows the **6th row** *"DA + IDA1 + DE_FCR …
      (forecast-driven realistic)"* alongside the co-opt ceiling row.
- [ ] `realistic ≤ ceiling` (the realistic bar/row never exceeds the ceiling).
- [ ] **Forecast-effect gap panel** renders: `full gap = forecast effect +
      timing cost`, with **Forecast effect** signed (negative = forecast helped).
- [ ] The panel caption states DA stays a full-power financial commitment and
      only physical IDA execution competes with reserve headroom.
- [ ] **Download** the Excel and confirm the `Assumptions` sheet carries the
      reserve-first / walk-forward / capacity-headroom-only / no-activation rows.

## 4. Verify Data Trust

Open the **Data Trust** tab:

- [ ] The **coverage matrix** lists DE_LU with `DA` coverage%, `IDA1` =
      `Synthetic demo (N)`, and `Reserve (capacity)` = `FCR`.
- [ ] The IDA source sub-table labels DE_LU/1 as `Synthetic demo`.

## 5. Capture screenshots

Capture: the strategy-comparison bar+table, the forecast-effect gap panel, the
Data Trust coverage matrix, and the export's `Assumptions` sheet. Store them with
the increment's PR or review notes.

## 6. Clean up

```bash
python scripts/seed_demo_9_2b.py --clean    # drops the demo cache tables + sample CSV
```

`--clean` only removes tables marked as `Synthetic demo` by the seed script. Use
`--force --clean` only if you intentionally want to remove unmarked DE_LU cache
tables.
