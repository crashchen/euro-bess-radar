import pandas as pd
import streamlit as st

from src.pages.forward_scenarios import _benchmark_comparison_figure
from src.trader_benchmark import MODEL_REVENUE_COLUMN, QUOTE_REVENUE_COLUMN

st.set_page_config(layout="wide")
comparison = pd.DataFrame({
    "year": [2027, 2028],
    QUOTE_REVENUE_COLUMN: [123456, 145678],
    MODEL_REVENUE_COLUMN: [190000, 193512],
})
st.plotly_chart(
    _benchmark_comparison_figure(comparison, chart_template="plotly_dark"),
    width="stretch",
)
