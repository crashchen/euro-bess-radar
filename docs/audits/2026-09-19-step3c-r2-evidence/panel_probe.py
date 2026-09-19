"""AppTest the actual imported-data sections and record their eight outputs.

Run from outside either checkout, setting PYTHONPATH to the desired checkout:
  PYTHONPATH=/path/to/checkout /path/to/.venv/bin/python panel_probe.py output.json
Compare the two JSON files' metrics/expanders, ignoring the source path.
"""

import json
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.pages import simulation_cockpit

at = AppTest.from_file(str(Path(__file__).with_name("panel_harness.py"))).run(
    timeout=30,
)
result = {
    "source": simulation_cockpit.__file__,
    "exceptions": [item.message for item in at.exception],
    "metrics": [
        {"label": item.label, "value": item.value} for item in at.metric
    ],
    "expanders": [item.label for item in at.expander],
}
assert not result["exceptions"], result["exceptions"]
expected = [
    ("MAE", "EUR 123.45/MW/h"),
    ("Bias", "EUR +0.00/MW/h"),
    ("RMSE", "EUR 123.45/MW/h"),
    ("Skill vs flat mean", "+15%"),
    ("Activation-energy overlay", "EUR 123,456,789"),
    ("Capture share", "1.0%"),
    ("Imbalance settlement overlay", "EUR -98,765,432"),
    ("Capture share", "1.0%"),
]
assert [(x["label"], x["value"]) for x in result["metrics"]] == expected
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
print(f"{simulation_cockpit.__file__}: eight expected metrics; no exceptions")
