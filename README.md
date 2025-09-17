# 🍺 Berlin Project - Sales Data Analysis System

A comprehensive Python-based sales data analysis system for bar operations, designed to provide actionable business insights.



## 🚀 Features

- **📊 Exploratory Data Analysis (EDA)**: Comprehensive analysis of sales data with automated insights
- **📈 Performance Metrics**: Calculate key business metrics, KPIs, and trend analysis
- **🎨 Advanced Visualizations**: Generate detailed charts, graphs, and interactive dashboards
- **🔍 Business Intelligence**: Identify patterns, opportunities, and optimization strategies
- **📋 Automated Reporting**: Generate comprehensive analysis reports with actionable recommendations
- **⚡ Clean Code**: Well-documented and maintainable codebase
- **🍺 Category-Specific Analysis**: Specialized analysis for Beer, Signature Cocktails, and Happy Hour

## 📋 Requirements

- Python 3.8+
- pandas >= 1.5.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/berlin-project.git
cd berlin-project
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

### Quick Start

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

### Command Line Usage

```bash
python sales_analyzer.py
```

## 📁 Project Structure

```
berlin-project/
├── sales_analyzer.py              # Main analysis module
├── requirements.txt               # Python dependencies
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
└── data/                         # Data directory (not tracked)
    └── sample_data_structure.md  # Data format documentation
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

## 🎯 Analysis Features

### 📊 Basic Metrics
- Total sales and units sold
- Average ticket size and pricing
- Product count and performance analysis

### 🏷️ Category Analysis
- Performance by product category
- Menu section analysis
- Price distribution and range analysis

### 🍺 Specialized Analysis
- **Beer Sales Analysis**: Detailed beer performance metrics
- **Signature Cocktails**: Premium cocktail analysis
- **Happy Hour**: Promotional period performance

### 🔍 Business Intelligence
- Top performing products identification
- High rotation, low price product analysis
- Premium popular product insights
- Low performance product identification

### 📈 Visualizations
- Sales distribution charts
- Top products analysis
- Price vs quantity relationships
- Category performance graphs
- Specialized category breakdowns

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