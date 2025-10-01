"""
Multi-Dashboard Analysis for Berlin Project
==========================================

This module creates multiple focused dashboards to analyze sales data 
from a bar/restaurant in Melbourne, Australia.

Purpose:
--------
- Replaces a single dense dashboard with 4 thematic, readable dashboards
- Automatically detects which months have available data (dynamic)
- Generates professional visualizations for business analysis
- Considers Melbourne seasons (southern hemisphere)

Generated Dashboards:
--------------------
1. Temporal Trends: Monthly trends, growth, seasonality
2. Product Performance: Top products, consistency, Pareto analysis
3. Category Analysis: Beverages vs food, menu sections, pricing
4. Business Metrics: KPIs, average transactions, normalized metrics

Typical Usage:
-------------
    analyzer = MultiDashboardAnalyzer()
    analyzer.run_all_dashboards()

Author: Daniel Montes
Date: 2025-09-24
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class MultiDashboardAnalyzer:
    """
    Sales analyzer that generates multiple focused dashboards.
    
    This class is designed to analyze sales data from a food & beverage business,
    creating visualizations that are easy to read and understand for decision making.
    
    Key Features:
    -------------
    - Automatic detection of available months (no manual updates needed)
    - Considers Melbourne, Australia seasons (southern hemisphere)
    - Generates 4 thematic dashboards instead of 1 dense dashboard
    - Analysis of products, categories, trends, and business metrics
    
    Attributes:
    -----------
    monthly_data : Dict[str, pd.DataFrame]
        Data organized by month (key: month name, value: DataFrame)
    consolidated_data : Optional[pd.DataFrame]
        All data combined into a single DataFrame
    analysis_results : Dict[str, any]
        Cached analysis results for reuse
    available_months : List[str]
        List of months that have available data (automatically detected)
    
    Example:
    --------
    >>> analyzer = MultiDashboardAnalyzer()
    >>> analyzer.run_all_dashboards()
    # Generates 4 PNG files with thematic dashboards
    """
    
    def __init__(self):
        """
        Initialize the multi-dashboard analyzer.
        
        Sets up the necessary data structures to store:
        - Separated monthly data
        - Consolidated data from all months
        - Analysis results to avoid recalculations
        - List of months with available data
        """
        self.monthly_data: Dict[str, pd.DataFrame] = {}
        self.consolidated_data: Optional[pd.DataFrame] = None
        self.analysis_results: Dict[str, any] = {}
        self.available_months: List[str] = []
        
    def load_all_data(self) -> None:
        """
        Load all available monthly sales data from the reports directory.
        
        This method:
        1. Scans the 'reports' directory for month folders
        2. Loads CSV files with sales data from each month
        3. Cleans and standardizes the data format
        4. Adds calculated fields (seasons, price categories, etc.)
        5. Tracks which months have valid data (self.available_months)
        6. Creates a consolidated dataset combining all months
        
        Data Processing:
        ---------------
        - Skips the first row (header: "Item Sold Report")
        - Standardizes column names
        - Converts price/sales fields from strings to numbers
        - Adds Melbourne seasons (Summer: Dec-Feb, Autumn: Mar-May, etc.)
        - Categorizes products by price ranges (Budget, Mid-Range, Premium, Luxury)
        
        Raises:
        -------
        Exception
            If there are issues reading individual CSV files (logged but not fatal)
        
        Side Effects:
        ------------
        - Populates self.monthly_data with DataFrames for each month
        - Populates self.available_months with month names that have data
        - Creates self.consolidated_data with all data combined
        - Prints progress messages showing which months were loaded successfully
        """
        print("📊 Loading data for multi-dashboard analysis...")
        
        # Define the path to reports and all possible months
        reports_dir = Path("reports")
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        
        # Iterate through all possible months to find available data
        for month in months:
            month_dir = reports_dir / month
            csv_file = month_dir / "report-sales_takings-item_sold.csv"
            
            # Only process months that have data files
            if csv_file.exists():
                try:
                    # Load CSV, skipping first row which contains "Item Sold Report"
                    df = pd.read_csv(csv_file, skiprows=1)
                    
                    # Standardize column names to ensure consistency
                    df.columns = [
                        "Menu Section", "Menu Item", "Size", "Portion", 
                        "Category", "Unit Price", "Quantity", "Sales", "% of Sales"
                    ]
                    
                    # Clean numeric columns - remove $ signs and commas, convert to numbers
                    for col in ["Unit Price", "Quantity", "Sales"]:
                        df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Remove any rows where sales data is missing/invalid
                    df = df.dropna(subset=['Sales'])
                    
                    # Add temporal metadata for analysis
                    df['Month'] = month
                    df['Month_Number'] = months.index(month) + 1  # 1-12 for Jan-Dec
                    df['Quarter'] = (df['Month_Number'] - 1) // 3 + 1  # Q1-Q4
                    
                    # Add Melbourne seasons (Southern Hemisphere - opposite to Northern)
                    # Summer: Dec-Feb, Autumn: Mar-May, Winter: Jun-Aug, Spring: Sep-Nov
                    season_map = {1: 'Summer', 2: 'Summer', 3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
                                 6: 'Winter', 7: 'Winter', 8: 'Winter', 9: 'Spring', 10: 'Spring', 
                                 11: 'Spring', 12: 'Summer'}
                    df['Season'] = df['Month_Number'].map(season_map)
                    
                    # Categorize products by price for analysis
                    # Budget: $0-20, Mid-Range: $20-30, Premium: $30-50, Luxury: $50+
                    df['Price_Category'] = pd.cut(df['Unit Price'], 
                                                 bins=[0, 20, 30, 50, np.inf], 
                                                 labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
                    
                    # Store the processed data and track this month as available
                    self.monthly_data[month] = df
                    self.available_months.append(month)
                    print(f"    ✅ {month}: {len(df)} items")
                    
                except Exception as e:
                    print(f"    ❌ Error loading {month}: {e}")
        
        if self.monthly_data:
            self.consolidated_data = pd.concat(self.monthly_data.values(), ignore_index=True)
            print(f"📈 Total: {len(self.monthly_data)} months, {len(self.consolidated_data)} records")
            print(f"📅 Available months: {', '.join(self.available_months)}")

    def create_temporal_trends_dashboard(self) -> None:
        """
        Create Dashboard 1: Temporal Trends and Growth Analysis.
        
        This dashboard focuses on time-based analysis showing how sales evolve
        over the available months. It includes trend analysis, growth rates,
        and seasonal patterns specific to Melbourne, Australia.
        
        Charts Generated:
        ----------------
        1. Top-left: Monthly Sales Trend with trend line (shows overall direction)
        2. Top-right: Month-over-Month Growth Rate (green/red bars for growth/decline)
        3. Bottom-left: Sales vs Quantity Correlation (relationship analysis)
        4. Bottom-right: Seasonal Performance (Melbourne seasons with appropriate colors)
        
        Key Features:
        ------------
        - Dynamic month detection (uses self.available_months)
        - Trend line calculation with visual direction indicator
        - Growth rate calculation and color coding
        - Melbourne seasonal analysis (Summer, Autumn, Winter, Spring)
        - Value labels on charts for easy reading
        
        Output:
        -------
        Saves 'dashboard_1_temporal_trends.png' file and displays the dashboard
        
        Returns:
        --------
        None
        """
        print("📊 Creating Dashboard 1: Temporal Trends...")
        
        if self.consolidated_data is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dashboard 1: Temporal Trends & Growth Analysis', fontsize=18, fontweight='bold')
        
        # Prepare monthly data - DYNAMIC!
        months_order = self.available_months.copy()  # Use detected months
        monthly_sales = {}
        monthly_qty = {}
        
        for month in months_order:
            monthly_sales[month] = self.monthly_data[month]['Sales'].sum()
            monthly_qty[month] = self.monthly_data[month]['Quantity'].sum()
        
        months_short = [m[:3] for m in monthly_sales.keys()]
        sales_values = list(monthly_sales.values())
        qty_values = list(monthly_qty.values())
        
        # 1. Monthly Sales Trend with Trendline
        axes[0, 0].plot(months_short, sales_values, marker='o', linewidth=3, markersize=10, 
                       color='#2E86AB', label='Monthly Sales')
        axes[0, 0].fill_between(months_short, sales_values, alpha=0.3, color='#2E86AB')
        
        # Add trend line
        x_numeric = range(len(sales_values))
        z = np.polyfit(x_numeric, sales_values, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(months_short, p(x_numeric), "--", color='red', linewidth=2, 
                       label=f"Trend {'↗️' if z[0] > 0 else '↘️'}")
        
        axes[0, 0].set_title('Monthly Sales Trend', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Sales ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(sales_values):
            axes[0, 0].annotate(f'${v/1000:.0f}K', (i, v), textcoords="offset points", 
                               xytext=(0,10), ha='center', fontsize=9)
        
        # 2. Month-over-Month Growth Rate
        growth_rates = [((sales_values[i] - sales_values[i-1]) / sales_values[i-1] * 100) 
                       for i in range(1, len(sales_values))]
        growth_months = months_short[1:]
        colors = ['green' if x >= 0 else 'red' for x in growth_rates]
        
        bars = axes[0, 1].bar(growth_months, growth_rates, color=colors, alpha=0.7)
        axes[0, 1].set_title('Month-over-Month Growth Rate', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Growth Rate (%)')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, rate in zip(bars, growth_rates):
            height = bar.get_height()
            axes[0, 1].annotate(f'{rate:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3 if height >= 0 else -15),
                               textcoords="offset points", ha='center', va='bottom')
        
        # 3. Sales vs Quantity Correlation
        axes[1, 0].scatter(qty_values, sales_values, s=100, alpha=0.7, color='purple')
        axes[1, 0].plot(qty_values, sales_values, '--', alpha=0.5, color='purple')
        
        # Add month labels
        for i, month in enumerate(months_short):
            axes[1, 0].annotate(month, (qty_values[i], sales_values[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        axes[1, 0].set_title('Sales vs Quantity Relationship', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Total Quantity Sold')
        axes[1, 0].set_ylabel('Total Sales ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Seasonal Performance (Melbourne seasons)
        seasonal_data = self.consolidated_data.groupby('Season')['Sales'].sum()
        season_colors = {'Summer': '#FFD700', 'Autumn': '#FF8C00', 'Winter': '#4169E1', 'Spring': '#32CD32'}
        colors_list = [season_colors.get(season, 'gray') for season in seasonal_data.index]
        
        bars = axes[1, 1].bar(seasonal_data.index, seasonal_data.values, color=colors_list, alpha=0.8)
        axes[1, 1].set_title('Seasonal Performance (Melbourne)', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Total Sales ($)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].annotate(f'${height/1000:.0f}K', xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('dashboard_1_temporal_trends.png', dpi=300, bbox_inches='tight')
        print("✅ Dashboard 1 saved as 'dashboard_1_temporal_trends.png'")
        plt.show()

    def create_product_performance_dashboard(self) -> None:
        """
        Create Dashboard 2: Product Performance Analysis.
        
        This dashboard focuses on identifying top-performing products and their
        sales contribution. Simplified to show only the most actionable insights.
        
        Charts Generated:
        ----------------
        1. Left: Top 10 Products by Sales (horizontal bar chart)
        2. Right: Sales Concentration (Pareto analysis - 80/20 rule)
        
        Key Features:
        ------------
        - Dynamic month detection (uses self.available_months)
        - Value labels on all charts for easy reading
        - Pareto analysis to identify critical products (80/20 rule)
        - Clean, focused layout for better readability
        
        Output:
        -------
        Saves 'dashboard_2_product_performance.png' file and displays the dashboard
        
        Returns:
        --------
        None
        """
        print("📊 Creating Dashboard 2: Product Performance...")
        
        if self.consolidated_data is None:
            return
        
        # Create figure with proper gridspec from start
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Dashboard 2: Product Performance Analysis', fontsize=18, fontweight='bold', y=0.95)
        
        # Create gridspec with proper spacing
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25, height_ratios=[1.3, 1], 
                             top=0.88, bottom=0.08, left=0.08, right=0.95)
        
        # 1. Top 10 Products by Sales (spanning top row)
        top_products = self.consolidated_data.groupby('Menu Item')['Sales'].sum().nlargest(10)
        
        # Create a single axes spanning the top row
        ax_top = fig.add_subplot(gs[0, :])
        ax_top.barh(range(len(top_products)), top_products.values, color='coral')
        ax_top.set_yticks(range(len(top_products)))
        ax_top.set_yticklabels([p[:30] + '...' if len(p) > 30 else p for p in top_products.index], fontsize=10)
        ax_top.set_title('Top 10 Products by Sales', fontsize=14, fontweight='bold', pad=10)
        ax_top.set_xlabel('Sales ($)', fontsize=11)
        ax_top.grid(True, alpha=0.3, axis='x')
        
        # Invert y-axis so highest sales are at the top
        ax_top.invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(top_products.values):
            ax_top.annotate(f'${v/1000:.0f}K', (v, i), xytext=(5, 0), 
                               textcoords='offset points', va='center', fontsize=9)
        
        # 2. Sales Concentration (Pareto Analysis) - Bottom left
        ax_bottom_left = fig.add_subplot(gs[1, 0])
        product_sales = self.consolidated_data.groupby('Menu Item')['Sales'].sum().sort_values(ascending=False)
        cumulative_pct = (product_sales.cumsum() / product_sales.sum() * 100)
        
        ax_bottom_left.plot(range(1, min(51, len(cumulative_pct) + 1)), 
                       cumulative_pct.iloc[:50].values, linewidth=3, color='darkblue')
        ax_bottom_left.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Rule')
        ax_bottom_left.set_title('Sales Concentration (Pareto)', fontsize=14, fontweight='bold')
        ax_bottom_left.set_xlabel('Top N Products')
        ax_bottom_left.set_ylabel('Cumulative Sales %')
        ax_bottom_left.grid(True, alpha=0.3)
        ax_bottom_left.legend()
        
        # Find 80% point
        products_80_pct = (cumulative_pct <= 80).sum()
        ax_bottom_left.annotate(f'{products_80_pct} products = 80% sales', 
                           xy=(products_80_pct, 80), xytext=(products_80_pct + 5, 70),
                           arrowprops=dict(arrowstyle='->', color='red'), fontsize=10)
        
        # 3. Sales per Item Distribution - Bottom right
        ax_bottom_right = fig.add_subplot(gs[1, 1])
        sales_per_item = self.consolidated_data['Sales'] / self.consolidated_data['Quantity'].replace(0, 1)
        
        ax_bottom_right.hist(sales_per_item[sales_per_item < 100], bins=25, color='lightcoral', 
                       alpha=0.7, edgecolor='black')
        ax_bottom_right.set_title('Sales per Item Distribution', fontsize=14, fontweight='bold')
        ax_bottom_right.set_xlabel('Sales per Item ($)')
        ax_bottom_right.set_ylabel('Frequency')
        ax_bottom_right.grid(True, alpha=0.3)
        
        # Add statistics
        mean_sales = sales_per_item.mean()
        median_sales = sales_per_item.median()
        ax_bottom_right.axvline(mean_sales, color='red', linestyle='--', label=f'Mean: ${mean_sales:.2f}')
        ax_bottom_right.axvline(median_sales, color='blue', linestyle='--', label=f'Median: ${median_sales:.2f}')
        ax_bottom_right.legend()
        
        plt.savefig('dashboard_2_product_performance.png', dpi=300, bbox_inches='tight')
        print("✅ Dashboard 2 saved as 'dashboard_2_product_performance.png'")
        plt.show()

    def create_category_analysis_dashboard(self) -> None:
        """Dashboard 3: Category and Menu Section Analysis"""
        print("📊 Creating Dashboard 3: Category Analysis...")
        
        if self.consolidated_data is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dashboard 3: Category & Menu Section Analysis', fontsize=18, fontweight='bold')
        
        # 1. Beverage vs Food Trends - DYNAMIC!
        monthly_category = {}
        
        for month in self.available_months:
            df = self.monthly_data[month]
            category_sales = df.groupby('Category')['Sales'].sum()
            monthly_category[month] = {
                'Beverage': category_sales.get('Beverage', 0),
                'Food': category_sales.get('Food', 0)
            }
        
        months_short = [m[:3] for m in monthly_category.keys()]
        bev_sales = [monthly_category[m]['Beverage'] for m in monthly_category.keys()]
        food_sales = [monthly_category[m]['Food'] for m in monthly_category.keys()]
        
        axes[0, 0].plot(months_short, bev_sales, marker='o', label='Beverages', 
                       linewidth=3, color='#F18F01', markersize=8)
        axes[0, 0].plot(months_short, food_sales, marker='s', label='Food', 
                       linewidth=3, color='#C73E1D', markersize=8)
        axes[0, 0].set_title('Beverage vs Food Trends', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Sales ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Top Menu Sections
        section_sales = self.consolidated_data.groupby('Menu Section')['Sales'].sum().nlargest(10)
        
        axes[0, 1].barh(range(len(section_sales)), section_sales.values, color='lightgreen')
        axes[0, 1].set_yticks(range(len(section_sales)))
        axes[0, 1].set_yticklabels([s[:20] + '...' if len(s) > 20 else s for s in section_sales.index], fontsize=9)
        axes[0, 1].set_title('Top 10 Menu Sections', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Total Sales ($)')
        
        # Add value labels
        for i, v in enumerate(section_sales.values):
            axes[0, 1].annotate(f'${v/1000:.0f}K', (v, i), xytext=(5, 0), 
                               textcoords='offset points', va='center', fontsize=9)
        
        # 3. Price Segment Distribution (with legend and value ranges)
        price_dist = self.consolidated_data.groupby('Price_Category')['Sales'].sum()
        # Define ranges for each price category (matching the actual bins)
        price_ranges = {
            'Budget': 'Budget ($0-20)',
            'Mid-Range': 'Mid-Range ($20-30)',
            'Premium': 'Premium ($30-50)',
            'Luxury': 'Luxury ($50+)',
        }
        price_colors = {'Budget': 'green', 'Mid-Range': 'blue', 'Premium': 'orange', 'Luxury': 'red'}
        # Prepare legend labels with value ranges and actual sales values
        legend_labels = []
        for cat in price_dist.index:
            range_label = price_ranges.get(cat, cat)
            sales_value = price_dist[cat]
            legend_labels.append(f"{range_label}: \\${sales_value:,.0f}")
        colors_list = [price_colors.get(cat, 'gray') for cat in price_dist.index]

        # Pie chart without labels, legend will show all info
        wedges, texts, autotexts = axes[1, 0].pie(price_dist.values, labels=None, 
                                                 autopct='%1.1f%%', colors=colors_list, startangle=90)
        axes[1, 0].set_title('Sales by Price Segment', fontsize=14, fontweight='bold')
        # Add legend with value ranges and sales values
        import matplotlib
        # Temporarily disable LaTeX rendering for this legend
        matplotlib.rcParams['text.usetex'] = False
        axes[1, 0].legend(wedges, legend_labels, title="Price Segments", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, title_fontsize=11)
        # Make percentage text larger
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        # 4. Category Performance Heatmap
        section_monthly = self.consolidated_data.groupby(['Month', 'Menu Section'])['Sales'].sum().unstack(fill_value=0)
        top_sections = self.consolidated_data.groupby('Menu Section')['Sales'].sum().nlargest(8).index
        
        # Order months chronologically
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        available_month_order = [month for month in month_order if month in section_monthly.index]
        section_monthly = section_monthly.reindex(available_month_order)
        
        if len(top_sections) > 0:
            heatmap_data = section_monthly[top_sections].T
            sns.heatmap(heatmap_data, annot=False, cmap='YlOrRd', ax=axes[1, 1], 
                       cbar_kws={'label': 'Sales ($)'})
            axes[1, 1].set_title('Top Sections: Monthly Performance', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Menu Section')
            axes[1, 1].tick_params(axis='y', labelsize=8)
        
        plt.tight_layout()
        plt.savefig('dashboard_3_category_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Dashboard 3 saved as 'dashboard_3_category_analysis.png'")
        plt.show()

    def create_business_metrics_dashboard(self) -> None:
        """Dashboard 4: Category Performance Analysis"""
        print("📊 Creating Dashboard 4: Category Performance...")
        
        if self.consolidated_data is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dashboard 4: Category Performance Analysis', fontsize=18, fontweight='bold')
        
        # Order months chronologically for all graphs
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        available_month_order = [month for month in month_order if month in self.available_months]
        months_short = [m[:3] for m in available_month_order]
        
        # 1. Top 5 Products Monthly Performance - DYNAMIC!
        # Get top 10 products overall, but show only top 5 to avoid clutter
        top_10_products = self.consolidated_data.groupby('Menu Item')['Sales'].sum().nlargest(10).index
        top_5_products = top_10_products[:5]
        
        # Colors for different products
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # Plot monthly performance for each top product
        for i, product in enumerate(top_5_products):
            monthly_sales = []
            for month in available_month_order:
                month_data = self.monthly_data[month]
                product_sales = month_data[month_data['Menu Item'] == product]['Sales'].sum()
                monthly_sales.append(product_sales)
            
            # Truncate product name for legend
            display_name = product[:20] + "..." if len(product) > 20 else product
            axes[0, 0].plot(months_short, monthly_sales, marker='o', linewidth=2, 
                           label=display_name, color=colors[i % len(colors)], markersize=6)
        
        axes[0, 0].set_title('Top 5 Products Monthly Performance', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Monthly Sales ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 2. Top Premium Cocktail Products Monthly Performance - DYNAMIC!
        # Filter for premium cocktail products - specifically "Premium Classics B"
        cocktail_data = self.consolidated_data[
            self.consolidated_data['Menu Section'] == 'Premium Classics B'
        ]
        
        if not cocktail_data.empty:
            # Get top 5 premium cocktail products
            top_5_cocktails = cocktail_data.groupby('Menu Item')['Sales'].sum().nlargest(5).index
            
            # Colors for different cocktail products (elegant cocktail colors)
            cocktail_colors = ['#8A2BE2', '#FF69B4', '#00CED1', '#FFD700', '#FF6347']
            
            for i, cocktail in enumerate(top_5_cocktails):
                cocktail_monthly = []
                for month in available_month_order:
                    month_data = self.monthly_data[month]
                    month_cocktail_sales = month_data[month_data['Menu Item'] == cocktail]['Sales'].sum()
                    cocktail_monthly.append(month_cocktail_sales)
                
                # Truncate cocktail name for legend
                display_name = cocktail[:18] + "..." if len(cocktail) > 18 else cocktail
                axes[0, 1].plot(months_short, cocktail_monthly, marker='d', linewidth=2, 
                               label=display_name, color=cocktail_colors[i % len(cocktail_colors)], markersize=6)
        
        axes[0, 1].set_title('Top 5 Premium Cocktails Monthly Performance', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Monthly Sales ($)')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Top Beer Products Monthly Performance - DYNAMIC!
        # Filter for beer products - only "Beer Berlin" section and exclude combo drinks
        beer_data = self.consolidated_data[
            (self.consolidated_data['Menu Section'] == 'Beer Berlin') &
            (~self.consolidated_data['Menu Item'].str.contains('\\+', case=False, na=False))
        ]
        
        if not beer_data.empty:
            # Get top 5 beer products
            top_5_beers = beer_data.groupby('Menu Item')['Sales'].sum().nlargest(5).index
            
            # Colors for different beer products
            beer_colors = ['#FFA500', '#8B4513', '#FFD700', '#CD853F', '#F4A460']
            
            for i, beer in enumerate(top_5_beers):
                beer_monthly = []
                for month in available_month_order:
                    month_data = self.monthly_data[month]
                    month_beer_sales = month_data[month_data['Menu Item'] == beer]['Sales'].sum()
                    beer_monthly.append(month_beer_sales)
                
                # Truncate beer name for legend
                display_name = beer[:18] + "..." if len(beer) > 18 else beer
                axes[1, 0].plot(months_short, beer_monthly, marker='s', linewidth=2, 
                               label=display_name, color=beer_colors[i % len(beer_colors)], markersize=6)
        
        axes[1, 0].set_title('Top 5 Beer Products Monthly Performance', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Monthly Sales ($)')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Top Food Products Monthly Performance - DYNAMIC!
        # Filter for food products - only "Food Berlin" section
        food_data = self.consolidated_data[
            self.consolidated_data['Menu Section'] == 'Food Berlin'
        ]
        
        if not food_data.empty:
            # Get top 5 food products
            top_5_foods = food_data.groupby('Menu Item')['Sales'].sum().nlargest(5).index
            
            # Colors for different food products (warm food colors)
            food_colors = ['#FF6347', '#FF8C00', '#32CD32', '#FFD700', '#FF69B4']
            
            for i, food in enumerate(top_5_foods):
                food_monthly = []
                for month in available_month_order:
                    month_data = self.monthly_data[month]
                    month_food_sales = month_data[month_data['Menu Item'] == food]['Sales'].sum()
                    food_monthly.append(month_food_sales)
                
                # Truncate food name for legend
                display_name = food[:18] + "..." if len(food) > 18 else food
                axes[1, 1].plot(months_short, food_monthly, marker='^', linewidth=2, 
                               label=display_name, color=food_colors[i % len(food_colors)], markersize=6)
        
        axes[1, 1].set_title('Top 5 Food Products Monthly Performance', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Monthly Sales ($)')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('dashboard_4_category_performance.png', dpi=300, bbox_inches='tight')
        print("✅ Dashboard 4 saved as 'dashboard_4_category_performance.png'")
        plt.show()

    def run_all_dashboards(self) -> None:
        """
        Execute the complete dashboard analysis workflow.
        
        This is the main method that orchestrates the entire analysis process.
        It loads data, validates it, and creates all 4 thematic dashboards.
        
        Workflow:
        ---------
        1. Load all available monthly data from reports directory
        2. Validate that data was found
        3. Create 4 focused dashboards:
           - Dashboard 1: Temporal Trends & Growth
           - Dashboard 2: Product Performance
           - Dashboard 3: Category Analysis  
           - Dashboard 4: Business Metrics
        4. Display summary of generated files and data period
        
        Output Files:
        ------------
        - dashboard_1_temporal_trends.png
        - dashboard_2_product_performance.png
        - dashboard_3_category_analysis.png
        - dashboard_4_business_metrics.png
        
        Console Output:
        --------------
        - Progress messages for each step
        - List of successfully loaded months
        - Summary of generated dashboard files
        - Data period coverage information
        
        Returns:
        --------
        None
            Method produces files and console output only
        """
        print("🚀 Creating Multiple Focused Dashboards (Dynamic Version)")
        print("=" * 70)
        
        # Step 1: Load and process all available monthly data
        self.load_all_data()
        
        # Step 2: Validate that we have data to work with
        if not self.monthly_data:
            print("❌ No data found!")
            return
        
        # Step 3: Generate all 4 thematic dashboards
        # Each dashboard focuses on a specific aspect of the business
        self.create_temporal_trends_dashboard()      # Time-based analysis
        self.create_product_performance_dashboard()  # Product-focused analysis
        self.create_category_analysis_dashboard()    # Category and menu analysis
        self.create_business_metrics_dashboard()     # KPIs and business metrics
        
        # Step 4: Display completion summary
        print("\n🎉 All dashboards created successfully!")
        print("Generated files:")
        print("  📊 dashboard_1_temporal_trends.png - Trends & Growth")
        print("  📊 dashboard_2_product_performance.png - Product Analysis")
        print("  📊 dashboard_3_category_analysis.png - Categories & Sections")
        print("  📊 dashboard_4_category_performance.png - Category Performance")
        print(f"\n📅 Data period: {', '.join(self.available_months)}")


def main():
    """
    Main execution function for the multi-dashboard analysis.
    
    This function serves as the entry point when the script is run directly.
    It creates an analyzer instance and executes the complete dashboard workflow.
    
    Usage:
    ------
    Can be called directly from command line:
        python multi_dashboard_analysis.py
        
    Or imported and called from other scripts:
        from multi_dashboard_analysis import main
        main()
    
    Side Effects:
    ------------
    - Creates 4 PNG dashboard files in the current directory
    - Prints progress and summary information to console
    - Processes all available monthly data from reports/ directory
    
    Returns:
    --------
    None
    """
    # Create analyzer instance and run complete workflow
    analyzer = MultiDashboardAnalyzer()
    analyzer.run_all_dashboards()


# Script entry point - runs main() when script is executed directly
if __name__ == "__main__":
    main()
