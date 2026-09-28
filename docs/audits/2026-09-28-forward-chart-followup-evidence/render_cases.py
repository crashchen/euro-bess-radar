"""Write offline 390px Plotly pages for the Forward endpoint-label probe."""

import sys
from pathlib import Path

import pandas as pd

from src.pages.forward_scenarios import _benchmark_comparison_figure
from src.trader_benchmark import MODEL_REVENUE_COLUMN, QUOTE_REVENUE_COLUMN


def main() -> None:
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    for n_years in (17, 20, 26, 32):
        years = list(range(2027, 2027 + n_years))
        frame = pd.DataFrame({
            "year": years,
            QUOTE_REVENUE_COLUMN: [100000.0] * n_years,
            MODEL_REVENUE_COLUMN: [105000.0] * n_years,
        })
        figure = _benchmark_comparison_figure(frame, chart_template="plotly_white")
        (output / f"{n_years}.html").write_text(figure.to_html(
            full_html=True,
            include_plotlyjs=True,
            default_width="390px",
            default_height="420px",
            config={"displayModeBar": False},
        ))


if __name__ == "__main__":
    main()
