# Contributing

Thanks for your interest in improving `euro-bess-radar`.

## Before You Start

- Open an issue or start a discussion before large or cross-cutting changes.
- Keep pull requests focused. Small, reviewable changes are much easier to merge.
- Update docs when user-facing behavior or setup steps change. Start from the [current validation snapshot](docs/validation/current.md) and the relevant [model contract](docs/design/delivery-duration-v1.md); historical audit logs describe their own revisions.

## Development Workflow

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ruff pytest-cov
ruff check src/ app.py tests/
python -m pytest tests/ -m "not slow" -q  # fast development loop
python -m pytest tests/ -m slow -q        # solver-heavy checks
python -m pytest tests/ -v
streamlit run app.py
```

Requires Python 3.11+ and Streamlit >=1.55,<2.0. For a focused example, `python -m pytest tests/test_analytics.py::TestDailySpreads -v` names an existing class. Use `python -m pytest tests/ --collect-only -q` for collected cases: parameterized cases are not the same count as test functions, and selection counts are not proof of a test run.

The complete suite includes `slow`; the default two opt-in PDF chart-render skips require `BESS_PULSE_RUN_KALEIDO_TESTS=1` and a working Kaleido/Chrome environment to execute. A skipped check is not rendered-output evidence. Record commands, exact revision, executed pass/skip counts and relevant browser/export evidence in the current snapshot or a dated handoff. Do not promote old audit counts into current validation.

CI runs lint, syntax and the full suite with coverage on Python 3.13 for pushes to main and PRs to any base. Its separate Python 3.11 / Streamlit 1.55.0 compatibility check exercises the panel/chart/caption/cockpit subset selected from `test_step2_views.py`, `test_ui_theme.py` and `test_market_grid_guards.py`. It pins those two runtime versions only; all other dependencies resolve from manifest ranges. Neither job is a full minimum-dependency test or live-provider/browser acceptance. The exact commands are in [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Pull Request Expectations

- Add or update tests for behavior changes.
- Keep public interfaces, exported files, and dashboard behavior clearly documented.
- Prefer incremental changes over broad refactors unless the refactor is the point of the PR.
- Preserve source-grid guards, signed cash, failure accounting, information-set boundaries and Project Case schema/fingerprints. Economic basis changes need their own contract and meaningful regression evidence.
- Report actual manual coverage using the [smoke checklist](docs/runbooks/manual-ui-smoke.md); AppTest text assertions and a checklist entry do not establish browser readability. Include viewport/sidebar state and rendered-export evidence when those surfaces change.
- Keep historical handoffs and frozen patches unchanged. Add a dated follow-up instead of overwriting evidence. A synthetic export used as deliberate audit evidence may be committed with its reproduction script and provenance; ordinary generated exports remain local.
- Review external-code work at its exact head and keep both CI checks green. The user invokes CC separately; agents must not automatically launch, resume, retry or schedule it. A draft approval does not authorize an automatic merge.

## Secrets, Local Files, and Generated Data

Do not commit:

- `.env` or API keys
- local virtual environments
- cache databases or cached market data
- generated exports such as ad hoc `.xlsx` or `.csv` reports
- machine-specific paths or workstation-specific notes

## Data Usage

This repository contains code, not redistributed market data. If your contribution touches external data ingestion or exports, keep source-specific terms, attribution requirements, and usage restrictions intact.
