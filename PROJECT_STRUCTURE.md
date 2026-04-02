# Berlin Project - Clean Structure

## 📁 Current Structure

```
berlin-project/
├── multi_year_sales_predictor.py   # Multi-year ML forecasting (2023-2025)
├── current_year_dashboard.py       # Year-specific dashboard generator (5 dashboards)
├── single_month_analyzer.py        # Single month deep-dive analysis
├── requirements.txt                # Python dependencies
├── README.md                       # Main documentation
├── PROJECT_STRUCTURE.md            # This file - structure guide
├── .gitignore                      # Git ignore rules
└── reports/                        # Data directory structure
    ├── 2023/                       # Full year 2023
    │   ├── January/
    │   │   └── report-sales_takings-item_sold.csv
    │   └── ...                     # All months
    ├── 2024/                       # Full year 2024
    │   └── ...                     # All months
    ├── 2025/                       # Full year 2025
    │   └── ...                     # All months
    └── 2026/                       # Current year (partial)
        └── ...                     # Available months
```

## 🎯 Usage

### Multi-Year Forecasting (Production ML)
```bash
python multi_year_sales_predictor.py
```
**Performance:** MAPE ~5%, R² ~0.29 (Gradient Boosting)

**Generates:**
- `multi_year_forecast.png` - 5-panel ML forecast dashboard
- `multi_year_forecast_report.txt` - Detailed forecast report with metrics

### Current Year Dashboard (2026)
```bash
python current_year_dashboard.py
```
**Generates:**
- `dashboard_1_temporal_trends_2026.png`
- `dashboard_2_product_performance_2026.png`
- `dashboard_3_category_analysis_2026.png`
- `dashboard_4_category_performance_2026.png`
- `dashboard_5_limited_edition_2026.png`

### Limited Edition Cocktail Tracker (Dashboard 5)
```python
from current_year_dashboard import main

# Without change markers
main(2026)

# Mark when you swapped the cocktail recipe
# → red dashed lines appear on all 4 charts in Dashboard 5
main(2026, cocktail_change_dates=["February", "May"])
```

### Specific Year Dashboard
```python
from current_year_dashboard import main
main(2024)  # Analyze 2024
```

### Single Month Analysis (Interactive)
```bash
python single_month_analyzer.py
```

**Interactive Prompt:**
```
Current date: October 2025
Press Enter to use current month or specify year and month.

Year [2025]: 2024
Month [October]: September

📊 Analyzing: September 2024
```

**Generated Files:**
- `sales_analysis_YEAR_MONTH.png` - General visualizations
- `category_analysis_YEAR_MONTH.png` - Category analysis
- `sales_report_YEAR_MONTH.txt` - Complete report

**Programmatic Usage:**
```python
from single_month_analyzer import main
analyzer = main(year=2024, month="April")  # Specific month
```

## 🧹 Changes Made

### Renamed Files:
- `advanced_sales_predictor.py` → `multi_year_sales_predictor.py`
- `sales_analyzer.py` → `single_month_analyzer.py`
- `multi_dashboard_analysis.py` → `current_year_dashboard.py`

### Updated Outputs:
- `advanced_sales_predictions.png` → `multi_year_forecast.png`
- `advanced_predictions_report.txt` → `multi_year_forecast_report.txt`
- `dashboard_*.png` → `dashboard_*_YEAR.png`
- Single month outputs now include year and month: `sales_analysis_YEAR_MONTH.png`

### Removed Files:
- Spanish documentation (GUIA_*, INICIO_*, INSTALACION.md, RESUMEN_*.md)
- Helper scripts (update_multi_dashboard.py, compare_predictions.py, quick_start.py)
- Legacy predictor (sales_predictor.py - didn't work with new structure)
- Old generated files

### Updated for New Structure:
- `current_year_dashboard.py` - Now reads from `reports/YEAR/MONTH/`
  - Accepts `year` parameter to analyze specific years
  - Generates year-specific output files
  - Updated docstrings and comments
- `single_month_analyzer.py` - Updated to work with new structure
  - Interactive mode: prompts user for year and month
  - Uses current month/year as default (just press Enter)
  - Accepts `year` and `month` parameters for programmatic use
  - Generates descriptive output files with year and month
  - Automatically builds correct path to CSV files

## ✅ Benefits

1. **Clearer Names**: File names now indicate their purpose
2. **Year-Specific**: Can analyze any year independently
3. **Organized Outputs**: Generated files include year in filename
4. **No Confusion**: Removed Spanish docs, kept only English
5. **Clean Structure**: Removed unnecessary helper scripts

## 📝 Notes

- All analysis tools now work with `reports/YEAR/MONTH/` structure
- Dashboard can analyze 2023, 2024, or 2025 independently
- Multi-year predictor uses all available years for forecasting
- Legacy `sales_predictor.py` kept for backward compatibility

