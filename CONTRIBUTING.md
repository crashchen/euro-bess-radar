# Contributing

Thanks for your interest in improving `euro-bess-radar`.

## Before You Start

- Open an issue or start a discussion before large or cross-cutting changes.
- Keep pull requests focused. Small, reviewable changes are much easier to merge.
- Update docs when user-facing behavior or setup steps change.

## Development Workflow

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v
streamlit run app.py
```

## Pull Request Expectations

- Add or update tests for behavior changes.
- Keep public interfaces, exported files, and dashboard behavior clearly documented.
- Prefer incremental changes over broad refactors unless the refactor is the point of the PR.

## Secrets, Local Files, and Generated Data

Do not commit:
- `.env` or API keys
- local virtual environments
- cache databases or cached market data
- generated exports such as ad hoc `.xlsx` or `.csv` reports
- machine-specific paths or workstation-specific notes

## Data Usage

This repository contains code, not redistributed market data. If your contribution touches external data ingestion or exports, keep source-specific terms, attribution requirements, and usage restrictions intact.
