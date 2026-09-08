"""Synthetic 0/1-quote uplift probe; run from the repo with PYTHONPATH=."""

import json

import pandas as pd

from src.analytics import calculate_intraday_uplift

index = pd.date_range("2025-09-01", periods=120, freq="h", tz="Europe/Berlin").tz_convert("UTC")
da = pd.DataFrame({"price_eur_mwh": 50.0}, index=index)
ida = pd.DataFrame({"intraday_price_eur_mwh": 60.0}, index=index)
results = {}
for source in ["da", "ida"]:
    target = da if source == "da" else ida
    sparse = target.drop(index[48:72])
    lone_quote = target.iloc[[48]].copy()
    lone_quote.iloc[0, 0] = 1_000_000.0
    singleton = pd.concat([sparse, lone_quote]).sort_index()
    for label, frame in [("absent", sparse), ("singleton", singleton)]:
        results[f"{source}_{label}"] = calculate_intraday_uplift(
            frame if source == "da" else da,
            frame if source == "ida" else ida,
            tz="Europe/Berlin",
        )
print(json.dumps(results, indent=2))
