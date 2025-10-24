# 🍺 Berlin Project - Advanced Sales Analytics System

A comprehensive Python-based sales data analysis system for bar operations, featuring multi-dashboard analytics and advanced business intelligence capabilities.

## 🚀 Key Features

### 🔮 **Multi-Year Sales Forecasting** (Production-Ready ML)
- **3 Years of Data**: Leverages 2023-2025 historical data for superior accuracy
- **3 ML Models**: XGBoost, Gradient Boosting (best), and Random Forest with auto-selection
- **Optimized Features**: Short-term lags, rolling statistics, cyclical encoding for maximum data utilization
- **Time Series CV**: Proper validation respecting temporal order (no data leakage)
- **Confidence Intervals**: Predictions with ±15% uncertainty ranges
- **5-Panel Dashboard**: Comprehensive visualizations including YoY comparison and seasonal patterns
- **Production Ready**: MAPE ~5%, R² ~0.29, iterative forecasting with dynamic lag updates

### 📊 **Current Year Dashboard System**
- **4 Specialized Dashboards**: Temporal Trends, Product Performance, Category Analysis, Category Performance
- **Year-Specific Analysis**: Analyze any year (2023, 2024, 2025) separately
- **Dynamic Data Loading**: Automatically processes monthly data from `reports/YEAR/MONTH/` structure
- **Professional Visualizations**: Publication-quality matplotlib layouts

### 🎯 **Advanced Business Intelligence**
- **Pareto Analysis (80/20 Rule)**: Identify products driving 80% of sales
- **Sales Concentration Analysis**: Understand revenue distribution patterns
- **Price Segmentation**: Budget, Mid-Range, Premium, and Luxury category analysis
- **Temporal Trend Analysis**: Month-over-month performance tracking with seasonal insights

### 🍺 **Category-Specific Analytics**
- **Beer Performance**: Top beer products with monthly tracking
- **Premium Cocktails**: Specialized analysis for Premium Classics menu section
- **Food Products**: Dedicated food category performance analysis
- **Happy Hour Analysis**: Promotional period effectiveness

## 📋 Requirements

- Python 3.8+
- pandas >= 1.5.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scipy >= 1.9.0
- scikit-learn >= 1.2.0
- numpy >= 1.21.0
- xgboost >= 1.7.0

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/DanielMontesD/berlinproject.git
cd berlinproject
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 📊 Usage

### 🔮 **Multi-Year Sales Forecasting** 

**ML-Based Forecasting (Recommended):**
```python
from multi_year_sales_predictor import AdvancedSalesPredictor

# Initialize predictor with multi-year data
predictor = AdvancedSalesPredictor()

# Run complete analysis (uses 2023-2025 data)
predictor.run_complete_analysis(
    years=[2023, 2024, 2025],  # Historical data years
    forecast_months=3           # Predict next 3 months
)
```

**Generated Files:**
- `multi_year_forecast.png` - Dashboard with 6 visualizations
- `multi_year_forecast_report.txt` - Detailed forecast report with YoY analysis

### 📊 **Current Year Dashboard**

```python
from current_year_dashboard import MultiDashboardAnalyzer

# Analyze current year (2025)
analyzer = MultiDashboardAnalyzer(year=2025)
analyzer.run_all_dashboards()

# Or analyze a different year
analyzer_2024 = MultiDashboardAnalyzer(year=2024)
analyzer_2024.run_all_dashboards()
```

**Generated Files:**
- `dashboard_1_temporal_trends_YEAR.png`
- `dashboard_2_product_performance_YEAR.png`
- `dashboard_3_category_analysis_YEAR.png`
- `dashboard_4_category_performance_YEAR.png`

### 📈 **Single Month Analysis**

**Interactive Mode (Recommended):**
```bash
python single_month_analyzer.py
```
The program will prompt you for:
- **Year** - Press Enter to use current year or type a year (e.g., 2024)
- **Month** - Press Enter to use current month or type a month (e.g., September)

**Programmatic Mode:**
```python
from single_month_analyzer import main

# Analyze specific month
analyzer = main(year=2024, month="September")
```

**Generated Files:**
- `sales_analysis_YEAR_MONTH.png` - General visualizations
- `category_analysis_YEAR_MONTH.png` - Category-specific analysis
- `sales_report_YEAR_MONTH.txt` - Complete report

### 🚀 **Command Line Usage**

```bash
# Multi-year sales forecasting (ML-based)
python multi_year_sales_predictor.py

# Current year dashboard (2025)
python current_year_dashboard.py

# Dashboard for specific year (e.g., 2024)
python -c "from current_year_dashboard import main; main(2024)"

# Single month analysis (interactive - uses current month by default)
python single_month_analyzer.py
```

## 📁 Project Structure

```
berlin-project/
├── multi_year_sales_predictor.py  # 🚀 Multi-year ML forecasting (2023-2025)
├── current_year_dashboard.py      # 📊 Year-specific dashboard generator
├── single_month_analyzer.py       # 📈 Single month deep-dive analysis
├── requirements.txt               # 📋 Python dependencies
├── .gitignore                     # 🔒 Git ignore rules
├── README.md                      # 📖 This documentation
├── PROJECT_STRUCTURE.md           # 📝 Project structure guide
└── data/                          # 📁 Sample data structure info
    └── sample_data_structure.md
```

### 🗂️ **Data Structure** (Confidential - Not Tracked)
```
reports/
├── 2023/                      # Full year 2023
│   ├── January/
│   │   └── report-sales_takings-item_sold.csv
│   ├── February/
│   │   └── report-sales_takings-item_sold.csv
│   └── ...                   # All 12 months
├── 2024/                      # Full year 2024
│   ├── January/
│   │   └── report-sales_takings-item_sold.csv
│   └── ...                   # All 12 months
└── 2025/                      # Current year (partial)
    ├── January/
    │   └── report-sales_takings-item_sold.csv
    └── ...                   # Available months
```

## 🔧 Getting Started

The project is designed to be simple and easy to use. Just follow the installation steps above and you'll be ready to analyze your sales data.

## 📈 Data Format

The system expects CSV files with the following structure:

| Column | Description | Example |
|--------|-------------|---------|
| Menu Section | Category section | "Beer", "Signature cocktails" |
| Menu Item | Product name | "Craft IPA", "Moscow Mule" |
| Size | Product size | "12oz", "Large" |
| Portion | Serving size | "1", "2" |
| Category | Product category | "Beverage", "Food" |
| Unit Price | Price per unit | "$5.50" |
| Quantity | Units sold | "25" |
| Sales | Total revenue | "$137.50" |
| % of Sales | Revenue percentage | "15.2%" |

## 🎯 Features Overview

### 🚀 **Multi-Year Sales Forecasting Features**
- **3 Years of Data**: Leverage 2023-2025 for superior pattern recognition (~30 months after preprocessing)
- **5-Panel Dashboard**: Sales forecast, YoY comparison, monthly patterns, model performance, seasonal heatmap
- **3 ML Models**: XGBoost, Gradient Boosting (primary), Random Forest with automatic selection
- **Optimized Features**: 16 engineered features including short-term lags (1-3), rolling stats (3, 6), cyclical encoding
- **Iterative Forecasting**: Each prediction uses previous predictions for dynamic lag features
- **Time Series CV**: 2-fold validation respecting temporal order (no data leakage)
- **Model Performance**: MAPE ~5%, R² ~0.29, RMSE ~$7,200 (production-validated)
- **Business Insights**: Automated recommendations based on historical patterns and YoY trends

### 📊 **Dashboard 1: Temporal Trends & Growth**
- **Monthly Sales Trends**: Track performance over time with seasonal insights (Melbourne seasons)
- **Sales vs Quantity Relationship**: Understand volume vs revenue patterns
- **Growth Rate Analysis**: Month-over-month growth tracking
- **Seasonal Performance**: Summer, Autumn, Winter, Spring analysis

### 🏆 **Dashboard 2: Product Performance Analysis**
- **Top 10 Products by Sales**: Identify your best performers with professional horizontal bar chart
- **Pareto Analysis (80/20 Rule)**: Discover which products drive 80% of your sales
- **Sales per Item Distribution**: Understand pricing and volume distribution patterns
- **Sales Concentration**: Visualize how sales are distributed across your product portfolio

### 🏷️ **Dashboard 3: Category Analysis & Segmentation**
- **Sales by Price Segment**: Budget ($0-20), Mid-Range ($20-30), Premium ($30-50), Luxury ($50+)
- **Category Performance Heatmap**: Monthly performance by menu section with chronological ordering
- **Menu Section Analysis**: Compare Beer Berlin, Food Berlin, Premium Classics, etc.
- **Price Distribution**: Understand your pricing strategy effectiveness

### 🍺 **Dashboard 4: Category Performance Analysis**
- **Top 5 Products Monthly Performance**: Track your best products over time
- **Top 5 Premium Cocktails**: Specialized analysis for Premium Classics B menu section
- **Top 5 Beer Products**: Dedicated beer performance tracking (Beer Berlin section only)
- **Top 5 Food Products**: Food category performance analysis (Food Berlin section)

### 🔍 **Advanced Analytics Features**
- **Dynamic Month Detection**: Automatically processes available months from reports/ directory
- **Melbourne Seasonal Analysis**: Southern Hemisphere seasons (Dec-Feb: Summer, Mar-May: Autumn, etc.)
- **Professional Visualizations**: Proper gridspec layouts, value labels, and clean formatting
- **Comprehensive Documentation**: Full English documentation with type hints and docstrings

## 🚨 Security Notice

**IMPORTANT**: This repository is configured to exclude confidential business data. The following files are automatically ignored:

- All CSV files (`*.csv`)
- Generated reports (`*.txt`, `*.png`, `*.pdf`)
- Virtual environments
- IDE configurations
- Log files

## 📝 Sample Data

To test the system, create a CSV file with the structure described above. The system will automatically process and analyze your data while maintaining confidentiality.

## 🤝 Contributing

This is a private project. For contribution guidelines, please contact the project maintainers.

## 📞 Support

For questions, support, or feature requests, please contact the development team.

## 📄 License

This project is proprietary and confidential. All rights reserved.

---

**Made with ❤️ for the Berlin Project Team**