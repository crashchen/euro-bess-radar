"""Read-only pytest collection/outcome recorder for a dated verification run."""
import ast
from collections import Counter
import json
import os
from pathlib import Path

ITEMS = []
REPORTS = []
WARNINGS = []


def pytest_collection_modifyitems(config, items):
    for item in items:
        path = Path(item.path).relative_to(config.rootpath).as_posix()
        ITEMS.append({"nodeid": item.nodeid, "file": path,
                      "callable": item.function.__qualname__,
                      "slow": item.get_closest_marker("slow") is not None})
    files = {}
    for path in sorted({i["file"] for i in ITEMS}):
        subset = [i for i in ITEMS if i["file"] == path]
        tree = ast.parse((config.rootpath / path).read_text())
        files[path] = {
            "test_function_definitions_ast": sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_") for n in ast.walk(tree)),
            "collected_callables": len({i["callable"] for i in subset}),
            "collected": len(subset),
            "slow": sum(i["slow"] for i in subset),
        }
    data = {"files": files, "items": ITEMS, "totals": {k:sum(v[k] for v in files.values()) for k in next(iter(files.values()))}}
    out = Path(os.environ["STEP4_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "test-inventory.json").write_text(json.dumps(data, indent=2)+"\n")


def pytest_runtest_logreport(report):
    if report.when == "call" or report.outcome != "passed":
        REPORTS.append({"nodeid":report.nodeid, "phase":report.when, "outcome":report.outcome})


def pytest_warning_recorded(warning_message, when, nodeid, location):
    WARNINGS.append({"category":warning_message.category.__name__, "message":str(warning_message.message),
                     "file":str(warning_message.filename), "line":warning_message.lineno, "phase":when, "nodeid":nodeid})


def pytest_sessionfinish(session, exitstatus):
    out = Path(os.environ["STEP4_OUTPUT"])
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter(r["outcome"] for r in REPORTS)
    data = {"exitstatus":int(exitstatus), "counts":dict(counts), "reports":REPORTS, "warnings":WARNINGS}
    (out / "run-results.json").write_text(json.dumps(data, indent=2)+"\n")
