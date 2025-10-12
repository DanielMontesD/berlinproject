# 🍺 Berlin Project - Advanced Sales Analytics System

A comprehensive Python-based sales data analysis system for bar operations, featuring multi-dashboard analytics and advanced business intelligence capabilities.

## 🚀 Key Features

### 🔮 **Predictive Analytics & Forecasting** ✨ NEW!
- **Monthly Sales Predictions**: Machine learning models to forecast future sales
- **Product Demand Forecasting**: Predict demand for top products with trend analysis
- **Seasonality Analysis**: Identify seasonal patterns (Melbourne seasons)
- **Multiple ML Models**: Linear Regression and Random Forest with automatic model selection
- **Actionable Insights**: Get business recommendations based on predictions

### 📊 **Multi-Dashboard Analytics System**
- **4 Specialized Dashboards**: Temporal Trends, Product Performance, Category Analysis, and Category Performance
- **Dynamic Data Loading**: Automatically processes monthly sales data from multiple periods
- **Professional Visualizations**: Advanced matplotlib layouts with proper scaling and spacing

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

### 🔮 **Predictive Analytics (NEW!)**

```python
from sales_predictor import SalesPredictor

# Initialize the sales predictor
predictor = SalesPredictor()

# Run complete prediction analysis
predictor.run_complete_analysis(
    months_ahead=3,      # Predict next 3 months
    top_products=10      # Analyze top 10 products
)

# Or run individual predictions
predictor.load_all_data()
predictor.predict_monthly_sales(months_ahead=3)
predictor.predict_product_demand(top_n=10)
predictor.analyze_seasonality()
predictor.create_prediction_visualizations()
predictor.generate_prediction_report("forecast_report.txt")
```

**Generated Files:**
- `sales_predictions.png` - Comprehensive prediction visualizations
- `sales_predictions_report.txt` - Detailed forecast report with insights

### 🎯 **Multi-Dashboard Analysis**

```python
from multi_dashboard_analysis import MultiDashboardAnalyzer

# Initialize the multi-dashboard analyzer
analyzer = MultiDashboardAnalyzer()

# Load all monthly data dynamically
analyzer.load_all_data()

# Generate all 4 dashboards
analyzer.run_all_dashboards()

# Or generate individual dashboards
analyzer.create_temporal_trends_dashboard()      # Dashboard 1: Trends & Growth
analyzer.create_product_performance_dashboard()  # Dashboard 2: Product Analysis  
analyzer.create_category_analysis_dashboard()    # Dashboard 3: Categories & Sections
analyzer.create_category_performance_dashboard() # Dashboard 4: Category Performance
```

### 📈 **Basic Sales Analysis**

```python
from sales_analyzer import SalesAnalyzer

# Initialize analyzer with your sales data
analyzer = SalesAnalyzer("your_sales_data.csv")

# Load and analyze data
analyzer.load_data()

# Generate comprehensive analysis
analyzer.calculate_basic_metrics()
analyzer.analyze_category_performance()
analyzer.get_top_products("Sales", 15)

# Create visualizations
analyzer.generate_visualizations("sales_analysis.png")
analyzer.generate_specific_category_analysis("category_analysis.png")

# Generate comprehensive report
analyzer.generate_comprehensive_report("report.txt")
```

### 🚀 **Command Line Usage**

```bash
# Run predictive analytics (NEW!)
python sales_predictor.py

# Run multi-dashboard analysis
python multi_dashboard_analysis.py

# Run basic sales analysis
python sales_analyzer.py
```

## 📁 Project Structure

```
berlin-project/
├── sales_predictor.py            # 🔮 Predictive analytics & forecasting (NEW!)
├── multi_dashboard_analysis.py   # 🎯 Multi-dashboard analytics system
├── sales_analyzer.py             # 📊 Basic sales analysis module
├── requirements.txt              # 📋 Python dependencies
├── .gitignore                   # 🔒 Git ignore rules (protects confidential data)
├── README.md                    # 📖 This documentation
└── data/                        # 📁 Data directory (not tracked)
    └── sample_data_structure.md # 📝 Data format documentation
```

### 🗂️ **Data Structure (Not Tracked)**
```
reports/                         # 📊 Monthly sales data (confidential)
├── January/
│   └── report-sales_takings-item_sold.csv
├── February/
│   └── report-sales_takings-item_sold.csv
├── March/
│   └── report-sales_takings-item_sold.csv
└── ...                         # Additional months
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

### 🔮 **Predictive Analytics Dashboard**
- **Monthly Sales Forecast**: Predict sales for the next 1-12 months using ML models
- **Product Demand Prediction**: Forecast demand for top products with trend indicators
- **Seasonal Performance**: Analyze seasonal patterns specific to Melbourne
- **Historical vs Predicted**: Visual comparison with confidence indicators
- **Model Accuracy Metrics**: MAE, RMSE, and R² scores for transparency
- **Business Insights**: Actionable recommendations based on predictions

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