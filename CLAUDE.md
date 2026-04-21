# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Activate the virtual environment before running anything:
```bash
source venv_berlin/Scripts/activate  # Windows (bash)
# or
venv_berlin\Scripts\activate.bat     # Windows (cmd)
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Scripts

```bash
# Multi-year ML sales forecasting (uses 2023–2025 data)
python multi_year_sales_predictor.py

# Year-specific dashboards (defaults to 2026)
python current_year_dashboard.py

# Single month deep-dive (interactive — prompts for year/month)
python single_month_analyzer.py
```

Programmatic usage:
```python
# Dashboard for a specific year
from current_year_dashboard import main
main(2024)

# Dashboard 5 with cocktail version-change markers
main(2026, cocktail_change_dates=["February", "June"])

# Single month analysis
from single_month_analyzer import main
analyzer = main(year=2024, month="September")

# ML forecasting
from multi_year_sales_predictor import AdvancedSalesPredictor
predictor = AdvancedSalesPredictor()
predictor.run_complete_analysis(years=[2023, 2024, 2025], forecast_months=3)
```

## Architecture

### Core Modules

**[utils.py](utils.py)** — shared constants and helpers imported by all other scripts:
- `MONTHS` list, `SEASON_MAP` (Melbourne/Southern Hemisphere seasons), `PRICE_BINS`/`PRICE_LABELS`
- `load_and_clean_sales_data(csv_path)` — standard CSV loader that skips the title row, standardizes column names, strips `$`/`%`, and drops null-sales rows
- `clean_currency(value)` — converts mixed `$X.XX` strings to float

**[current_year_dashboard.py](current_year_dashboard.py)** — `MultiDashboardAnalyzer` class:
- Instantiated with a `year` parameter; auto-discovers available months under `reports/YEAR/MONTH/`
- Generates 5 PNG dashboards: temporal trends, product performance, category analysis, category performance, and the limited edition cocktail tracker
- Dashboard 5 accepts `cocktail_change_dates` (list of month names) to draw red dashed version-change markers

**[multi_year_sales_predictor.py](multi_year_sales_predictor.py)** — `AdvancedSalesPredictor` class:
- Loads data across multiple years, engineers 16 temporal features (short-term lags 1–3, rolling stats at 3 and 6 months, cyclical encoding)
- Trains XGBoost, Gradient Boosting, and Random Forest; auto-selects best model by MAPE
- Uses `TimeSeriesSplit` (2-fold) to avoid data leakage
- Iterative forecasting: each predicted month feeds back as lag features for the next
- Outputs `multi_year_forecast.png` and `multi_year_forecast_report.txt`

**[single_month_analyzer.py](single_month_analyzer.py)** — single-month deep-dive:
- Interactive CLI (prompts for year/month, defaults to current month)
- Outputs `sales_analysis_YEAR_MONTH.png`, `category_analysis_YEAR_MONTH.png`, `sales_report_YEAR_MONTH.txt`

### Data Layout

CSV files are **not committed** (`.gitignore` excludes `*.csv`). Expected path pattern:
```
reports/
  YEAR/
    MonthName/
      report-sales_takings-item_sold.csv
```

Each CSV has a title row (skipped automatically) followed by:
`Menu Section, Menu Item, Size, Portion, Category, Unit Price, Quantity, Sales, % of Sales`

Prices use `$X.XX` format; percentages use `X.X%` format — `load_and_clean_sales_data` handles all cleaning.

### Output Files

Generated files (`.png`, `.txt`) are gitignored. Output filenames follow the pattern:
- `dashboard_1_temporal_trends_YEAR.png` … `dashboard_5_limited_edition_YEAR.png`
- `multi_year_forecast.png` / `multi_year_forecast_report.txt`
- `sales_analysis_YEAR_MONTH.png` / `category_analysis_YEAR_MONTH.png` / `sales_report_YEAR_MONTH.txt`

### Seasonal Logic

The project uses **Melbourne (Southern Hemisphere) seasons**:
- Summer: Dec–Feb | Autumn: Mar–May | Winter: Jun–Aug | Spring: Sep–Nov

This is intentional — do not change to Northern Hemisphere conventions.
