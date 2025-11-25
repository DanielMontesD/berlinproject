"""
Single Month Sales Analysis Module for Berlin Project
=====================================================

This module provides comprehensive analysis of sales data for a single month including:
- Exploratory data analysis (EDA)
- Performance metrics calculation
- Visualization generation
- Business insights extraction

Data Structure:
--------------
Expects data in: reports/YEAR/MONTH/report-sales_takings-item_sold.csv

Typical Usage:
-------------
    # Interactive mode (recommended) - uses current month by default
    python single_month_analyzer.py

    # The program will prompt:
    # Year [2025]: 2024
    # Month [October]: September

    # Programmatic mode
    from single_month_analyzer import main
    analyzer = main(year=2024, month="April")

Features:
--------
- Interactive prompts for year and month
- Automatic detection of current month/year as defaults
- Press Enter to use defaults (no typing needed for current month)
- Error handling with helpful messages if data not found

Author: Daniel Montes
Date: 2025-10-15
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import warnings
from pathlib import Path
from utils import (
    PRICE_BINS,
    PRICE_LABELS,
    load_and_clean_sales_data,
)

warnings.filterwarnings("ignore")

# Configure plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class SalesAnalyzer:
    """
    A comprehensive sales data analyzer for restaurant/bar operations.

    This class provides methods to load, clean, analyze, and visualize sales
    data with focus on business insights and performance metrics.

    Attributes:
        csv_file_path (Path): Path to the sales CSV file
        raw_data (Optional[pd.DataFrame]): Raw data loaded from CSV
        clean_data (Optional[pd.DataFrame]): Cleaned and processed data
        analysis_results (Dict[str, Any]): Cached analysis results
    """

    def __init__(self, csv_file_path: Union[str, Path]) -> None:
        """
        Initialize the SalesAnalyzer.

        Args:
            csv_file_path: Path to the sales CSV file

        Raises:
            FileNotFoundError: If the CSV file doesn't exist
        """
        self.csv_file_path = Path(csv_file_path)
        if not self.csv_file_path.exists():
            raise FileNotFoundError(f"Sales data file not found: {csv_file_path}")

        self.raw_data: Optional[pd.DataFrame] = None
        self.clean_data: Optional[pd.DataFrame] = None
        self.analysis_results: Dict[str, Any] = {}

    def load_data(self) -> pd.DataFrame:
        """
        Load and perform initial data cleaning.

        Returns:
            Cleaned DataFrame with sales data

        Raises:
            ValueError: If data cannot be loaded or is invalid
        """
        try:
            print(">> Loading sales data...")

            # Use shared utility to load and clean data
            self.clean_data = load_and_clean_sales_data(self.csv_file_path)
            self.raw_data = self.clean_data  # Keep compatibility if raw_data is accessed

            print(f"OK Data loaded: {len(self.clean_data)} products")
            print(f">> Available columns: {list(self.clean_data.columns)}")

            return self.clean_data

        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

    def _ensure_data_loaded(self) -> pd.DataFrame:
        """
        Ensure data is loaded, raise error if not available.

        Returns:
            Clean data DataFrame

        Raises:
            ValueError: If data cannot be loaded
        """
        if self.clean_data is None:
            self.load_data()

        if self.clean_data is None:
            raise ValueError("Failed to load data")

        return self.clean_data

    def calculate_basic_metrics(self) -> Dict[str, Union[float, int]]:
        """
        Calculate basic sales metrics.

        Returns:
            Dictionary containing key performance metrics
        """
        data = self._ensure_data_loaded()

        total_sales = float(data["Sales"].sum())
        total_units = int(data["Quantity"].sum())
        unique_products = len(data)
        avg_price = float(data["Unit Price"].mean())
        avg_ticket = total_sales / total_units if total_units > 0 else 0.0

        metrics = {
            "total_sales": total_sales,
            "total_units": total_units,
            "unique_products": unique_products,
            "average_price": avg_price,
            "average_ticket": avg_ticket,
        }

        self.analysis_results["basic_metrics"] = metrics
        return metrics

    def analyze_category_performance(self) -> pd.DataFrame:
        """
        Analyze sales performance by category.

        Returns:
            DataFrame with category performance metrics
        """
        data = self._ensure_data_loaded()

        category_stats = (
            data.groupby("Category")
            .agg(
                {
                    "Sales": "sum",
                    "Quantity": "sum",
                    "Menu Item": "count",
                    "Unit Price": "mean",
                }
            )
            .round(2)
        )

        category_stats.columns = [
            "Total_Sales",
            "Total_Units",
            "Unique_Products",
            "Average_Price",
        ]
        category_stats = category_stats.sort_values("Total_Sales", ascending=False)

        self.analysis_results["category_performance"] = category_stats
        return category_stats

    def get_top_products(self, metric: str = "Sales", top_n: int = 15) -> pd.DataFrame:
        """
        Get top performing products by specified metric.

        Args:
            metric: Metric to rank by ('Sales', 'Quantity', 'Unit Price')
            top_n: Number of top products to return

        Returns:
            DataFrame with top products

        Raises:
            ValueError: If metric is not valid
        """
        data = self._ensure_data_loaded()

        valid_metrics = ["Sales", "Quantity", "Unit Price"]
        if metric not in valid_metrics:
            raise ValueError(f"Metric must be one of: {valid_metrics}")

        top_products = data.nlargest(top_n, metric)[
            ["Menu Item", "Category", "Quantity", "Sales", "Unit Price", "% of Sales"]
        ]

        self.analysis_results[f"top_products_{metric.lower()}"] = top_products
        return top_products

    def analyze_menu_sections(self) -> pd.DataFrame:
        """
        Analyze performance by menu sections.

        Returns:
            DataFrame with menu section performance metrics
        """
        data = self._ensure_data_loaded()

        section_stats = (
            data.groupby("Menu Section")
            .agg(
                {
                    "Sales": "sum",
                    "Quantity": "sum",
                    "Menu Item": "count",
                    "Unit Price": "mean",
                }
            )
            .round(2)
        )

        section_stats.columns = [
            "Total_Sales",
            "Total_Units",
            "Unique_Products",
            "Average_Price",
        ]
        section_stats = section_stats.sort_values("Total_Sales", ascending=False)

        self.analysis_results["menu_sections"] = section_stats
        return section_stats

    def analyze_price_distribution(self) -> Dict[str, Any]:
        """
        Analyze price distribution and ranges.

        Returns:
            Dictionary with price analysis metrics
        """
        data = self._ensure_data_loaded()

        prices = data["Unit Price"]

        price_stats = {
            "min_price": float(prices.min()),
            "max_price": float(prices.max()),
            "mean_price": float(prices.mean()),
            "median_price": float(prices.median()),
            "std_price": float(prices.std()),
        }

        # Price range analysis using shared constants
        data["Price_Category"] = pd.cut(
            data["Unit Price"], bins=PRICE_BINS, labels=PRICE_LABELS
        )

        range_analysis: Dict[str, Dict[str, Union[int, float]]] = {}
        
        # Group by the new category
        for label in PRICE_LABELS:
            products_in_range = data[data["Price_Category"] == label]
            range_sales = products_in_range["Sales"].sum()
            range_analysis[label] = {
                "product_count": len(products_in_range),
                "total_sales": float(range_sales),
            }

        price_analysis = {"statistics": price_stats, "range_analysis": range_analysis}

        self.analysis_results["price_distribution"] = price_analysis
        return price_analysis

    def identify_business_patterns(
        self,
    ) -> Dict[str, List[Dict[str, Union[str, int, float]]]]:
        """
        Identify key business patterns and opportunities.

        Returns:
            Dictionary with identified patterns
        """
        data = self._ensure_data_loaded()

        patterns: Dict[str, List[Dict[str, Union[str, int, float]]]] = {
            "high_rotation_low_price": [],
            "premium_popular": [],
            "low_performance": [],
        }

        # High rotation, low price products
        high_rotation = data[data["Quantity"] >= 20]
        low_price_high_rotation = high_rotation[high_rotation["Unit Price"] <= 15]

        for _, row in low_price_high_rotation.iterrows():
            patterns["high_rotation_low_price"].append(
                {
                    "product": str(row["Menu Item"]),
                    "quantity": int(row["Quantity"]),
                    "price": float(row["Unit Price"]),
                    "sales": float(row["Sales"]),
                }
            )

        # Premium products with good sales
        premium = data[data["Unit Price"] >= 25]
        premium_popular = premium[premium["Quantity"] >= 5]

        for _, row in premium_popular.iterrows():
            patterns["premium_popular"].append(
                {
                    "product": str(row["Menu Item"]),
                    "quantity": int(row["Quantity"]),
                    "price": float(row["Unit Price"]),
                    "sales": float(row["Sales"]),
                }
            )

        # Low performance products
        low_performance = data[data["Quantity"] <= 2]

        for _, row in low_performance.iterrows():
            patterns["low_performance"].append(
                {
                    "product": str(row["Menu Item"]),
                    "quantity": int(row["Quantity"]),
                    "price": float(row["Unit Price"]),
                    "sales": float(row["Sales"]),
                }
            )

        self.analysis_results["business_patterns"] = patterns
        return patterns

    def generate_visualizations(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Generate comprehensive sales visualizations.

        Args:
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        data = self._ensure_data_loaded()

        print(">> Creating visualizations...")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle("Sales Analysis - Berlin Project", fontsize=16, fontweight="bold")

        # 1. Top 15 products by sales
        top_sales = data.nlargest(15, "Sales")
        axes[0, 0].barh(range(len(top_sales)), top_sales["Sales"], color="skyblue")
        axes[0, 0].set_yticks(range(len(top_sales)))
        axes[0, 0].set_yticklabels(
            [
                item[:25] + "..." if len(item) > 25 else item
                for item in top_sales["Menu Item"]
            ],
            fontsize=8,
        )
        axes[0, 0].set_xlabel("Sales ($)")
        axes[0, 0].set_title("Top 15 Products by Sales")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Sales distribution by category
        category_sales = data.groupby("Category")["Sales"].sum()
        axes[0, 1].pie(
            category_sales.values, labels=category_sales.index, autopct="%1.1f%%"
        )
        axes[0, 1].set_title("Sales Distribution by Category")

        # 3. Top 15 products by quantity
        top_quantity = data.nlargest(15, "Quantity")
        axes[0, 2].barh(
            range(len(top_quantity)), top_quantity["Quantity"], color="lightgreen"
        )
        axes[0, 2].set_yticks(range(len(top_quantity)))
        axes[0, 2].set_yticklabels(
            [
                item[:25] + "..." if len(item) > 25 else item
                for item in top_quantity["Menu Item"]
            ],
            fontsize=8,
        )
        axes[0, 2].set_xlabel("Quantity Sold")
        axes[0, 2].set_title("Top 15 Products by Quantity")
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Price vs Quantity relationship
        axes[1, 0].scatter(
            data["Unit Price"], data["Quantity"], alpha=0.6, color="orange"
        )
        axes[1, 0].set_xlabel("Unit Price ($)")
        axes[1, 0].set_ylabel("Quantity Sold")
        axes[1, 0].set_title("Price vs Quantity Relationship")
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Price distribution histogram
        axes[1, 1].hist(data["Unit Price"], bins=20, color="purple", alpha=0.7)
        axes[1, 1].set_xlabel("Unit Price ($)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Price Distribution")
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Sales by menu section
        section_sales = (
            data.groupby("Menu Section")["Sales"].sum().sort_values(ascending=True)
        )
        axes[1, 2].barh(range(len(section_sales)), section_sales.values, color="coral")
        axes[1, 2].set_yticks(range(len(section_sales)))
        axes[1, 2].set_yticklabels(
            [
                section[:20] + "..." if len(section) > 20 else section
                for section in section_sales.index
            ],
            fontsize=8,
        )
        axes[1, 2].set_xlabel("Sales ($)")
        axes[1, 2].set_title("Sales by Menu Section")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"OK Visualizations saved as '{save_path}'")

        plt.show()
        return fig

    def generate_specific_category_analysis(
        self, save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Generate specific analysis for Beer, Signature Cocktails, and Happy Hour.

        Args:
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        data = self._ensure_data_loaded()

        print(">> Creating specific category analysis...")

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "Specific Category Analysis - Berlin Project",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Beer Analysis
        beer_data = data[
            data["Menu Section"].str.contains("Beer", case=False, na=False)
        ]
        if len(beer_data) > 0:
            beer_sales = (
                beer_data.groupby("Menu Item")["Sales"]
                .sum()
                .sort_values(ascending=True)
            )
            axes[0].barh(
                range(len(beer_sales)), beer_sales.values, color="#FFA500", alpha=0.8
            )
            axes[0].set_yticks(range(len(beer_sales)))
            axes[0].set_yticklabels(
                [
                    item[:20] + "..." if len(item) > 20 else item
                    for item in beer_sales.index
                ],
                fontsize=9,
            )
            axes[0].set_xlabel("Sales ($)")
            axes[0].set_title("Beer Sales Analysis", fontweight="bold")
            axes[0].grid(True, alpha=0.3)

            # Add total sales text
            total_beer_sales = beer_sales.sum()
            axes[0].text(
                0.02,
                0.98,
                f"Total: ${total_beer_sales:,.2f}",
                transform=axes[0].transAxes,
                fontsize=10,
                fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
        else:
            axes[0].text(
                0.5,
                0.5,
                "No Beer Data Found",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
            axes[0].set_title("Beer Sales Analysis", fontweight="bold")

        # 2. Signature Cocktails Analysis
        signature_data = data[
            data["Menu Section"].str.contains(
                "Signature cocktails", case=False, na=False
            )
        ]
        if len(signature_data) > 0:
            signature_sales = (
                signature_data.groupby("Menu Item")["Sales"]
                .sum()
                .sort_values(ascending=True)
            )
            axes[1].barh(
                range(len(signature_sales)),
                signature_sales.values,
                color="#FF69B4",
                alpha=0.8,
            )
            axes[1].set_yticks(range(len(signature_sales)))
            axes[1].set_yticklabels(
                [
                    item[:20] + "..." if len(item) > 20 else item
                    for item in signature_sales.index
                ],
                fontsize=9,
            )
            axes[1].set_xlabel("Sales ($)")
            axes[1].set_title("Signature Cocktails Analysis", fontweight="bold")
            axes[1].grid(True, alpha=0.3)

            # Add total sales text
            total_signature_sales = signature_sales.sum()
            axes[1].text(
                0.02,
                0.98,
                f"Total: ${total_signature_sales:,.2f}",
                transform=axes[1].transAxes,
                fontsize=10,
                fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
        else:
            axes[1].text(
                0.5,
                0.5,
                "No Signature Cocktails Data Found",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_title("Signature Cocktails Analysis", fontweight="bold")

        # 3. Happy Hour Analysis
        happy_hour_data = data[
            data["Menu Section"].str.contains("Happy Hour", case=False, na=False)
        ]
        if len(happy_hour_data) > 0:
            happy_hour_sales = (
                happy_hour_data.groupby("Menu Item")["Sales"]
                .sum()
                .sort_values(ascending=True)
            )
            axes[2].barh(
                range(len(happy_hour_sales)),
                happy_hour_sales.values,
                color="#32CD32",
                alpha=0.8,
            )
            axes[2].set_yticks(range(len(happy_hour_sales)))
            axes[2].set_yticklabels(
                [
                    item[:20] + "..." if len(item) > 20 else item
                    for item in happy_hour_sales.index
                ],
                fontsize=9,
            )
            axes[2].set_xlabel("Sales ($)")
            axes[2].set_title("Happy Hour Analysis", fontweight="bold")
            axes[2].grid(True, alpha=0.3)

            # Add total sales text
            total_happy_hour_sales = happy_hour_sales.sum()
            axes[2].text(
                0.02,
                0.98,
                f"Total: ${total_happy_hour_sales:,.2f}",
                transform=axes[2].transAxes,
                fontsize=10,
                fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
        else:
            axes[2].text(
                0.5,
                0.5,
                "No Happy Hour Data Found",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
            )
            axes[2].set_title("Happy Hour Analysis", fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"OK Specific category analysis saved as '{save_path}'")

        plt.show()
        return fig

    def generate_comprehensive_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive analysis report.

        Args:
            output_path: Optional path to save the report

        Returns:
            Formatted report string
        """
        data = self._ensure_data_loaded()

        print(">> Generating comprehensive report...")

        # Calculate all metrics
        basic_metrics = self.calculate_basic_metrics()
        category_performance = self.analyze_category_performance()
        top_sales = self.get_top_products("Sales", 10)
        patterns = self.identify_business_patterns()

        # Generate report
        report_lines = [
            "=" * 80,
            "📊 COMPREHENSIVE SALES ANALYSIS REPORT - BERLIN PROJECT",
            "=" * 80,
            f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"📁 Data Source: {self.csv_file_path.name}",
            "",
            "📈 EXECUTIVE SUMMARY:",
            f"   • Total Sales: ${basic_metrics['total_sales']:,.2f}",
            f"   • Total Units Sold: {basic_metrics['total_units']:,}",
            f"   • Unique Products: {basic_metrics['unique_products']}",
            f"   • Average Ticket: ${basic_metrics['average_ticket']:.2f}",
            "",
            "🏆 TOP 10 PRODUCTS BY SALES:",
        ]

        for idx, row in top_sales.iterrows():
            report_lines.append(
                f"   {idx+1:2d}. {row['Menu Item']:<40} " f"${row['Sales']:>8.2f}"
            )

        report_lines.extend(
            [
                "",
                "📊 SALES BY CATEGORY:",
            ]
        )

        for category, stats in category_performance.iterrows():
            report_lines.append(
                f"   • {category:<15} ${stats['Total_Sales']:>8.2f} "
                f"({stats['Total_Units']:>3} units)"
            )

        report_lines.extend(
            [
                "",
                "🔍 KEY BUSINESS INSIGHTS:",
            ]
        )

        # Most sold product
        most_sold = data.loc[data["Quantity"].idxmax()]
        report_lines.append(
            f"   • Most Sold Product: {most_sold['Menu Item']} "
            f"({most_sold['Quantity']} units)"
        )

        # Highest revenue product
        highest_revenue = data.loc[data["Sales"].idxmax()]
        report_lines.append(
            f"   • Highest Revenue Product: {highest_revenue['Menu Item']} "
            f"(${highest_revenue['Sales']:.2f})"
        )

        # Dominant category
        dominant_category = category_performance.index[0]
        report_lines.append(f"   • Dominant Category: {dominant_category}")

        # Low performance count
        low_performance_count = len(patterns["low_performance"])
        report_lines.append(
            f"   • Low Performance Products (≤2 units): " f"{low_performance_count}"
        )

        report_lines.extend(
            [
                "",
                "🚀 HIGH ROTATION, LOW PRICE PRODUCTS:",
            ]
        )

        for product in patterns["high_rotation_low_price"][:5]:
            report_lines.append(
                f"   • {product['product']}: {product['quantity']} units "
                f"at ${product['price']:.2f}"
            )

        report_lines.extend(
            [
                "",
                "💎 PREMIUM POPULAR PRODUCTS:",
            ]
        )

        for product in patterns["premium_popular"][:5]:
            report_lines.append(
                f"   • {product['product']}: {product['quantity']} units "
                f"at ${product['price']:.2f}"
            )

        report_lines.extend(["", "=" * 80])

        report_text = "\n".join(report_lines)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"OK Report saved as '{output_path}'")

        return report_text


def main(year: int = 2025, month: str = "September") -> SalesAnalyzer:
    """
    Main function to execute comprehensive sales analysis for a specific month.

    This function demonstrates the complete workflow of the SalesAnalyzer class,
    including data loading, analysis, visualization, and report generation.

    Parameters:
    -----------
    year : int
        Year to analyze (default: 2025)
    month : str
        Month to analyze (default: "September")

    Returns:
    --------
    SalesAnalyzer instance with completed analysis

    Example Usage:
    -------------
    # Analyze September 2025
    analyzer = main(2025, "September")

    # Analyze April 2024
    analyzer = main(2024, "April")
    """
    print(f">> STARTING SALES ANALYSIS FOR {month} {year}")
    print("=" * 60)

    try:
        # Build path to CSV file with new structure
        csv_path = Path(f"reports/{year}/{month}/report-sales_takings-item_sold.csv")

        if not csv_path.exists():
            print(f"❌ ERROR: File not found: {csv_path}")
            print(f"\nAvailable structure should be:")
            print(f"   reports/{year}/{month}/report-sales_takings-item_sold.csv")
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        # Initialize analyzer
        analyzer = SalesAnalyzer(csv_path)

        # Load and analyze data
        analyzer.load_data()

        print("\n>> Executing comprehensive analysis...")

        # Generate all analyses
        analyzer.calculate_basic_metrics()
        analyzer.analyze_category_performance()
        analyzer.get_top_products("Sales", 15)
        analyzer.get_top_products("Quantity", 15)
        analyzer.analyze_menu_sections()
        analyzer.analyze_price_distribution()
        analyzer.identify_business_patterns()

        # Generate visualizations with descriptive names
        month_label = f"{year}_{month}"
        analyzer.generate_visualizations(Path(f"sales_analysis_{month_label}.png"))

        # Generate specific category analysis
        analyzer.generate_specific_category_analysis(
            Path(f"category_analysis_{month_label}.png")
        )

        # Generate comprehensive report
        analyzer.generate_comprehensive_report(Path(f"sales_report_{month_label}.txt"))

        print(f"\n>> ANALYSIS COMPLETED FOR {month} {year}!")
        print(">> Generated Files:")
        print(f"   • sales_analysis_{month_label}.png - General visualizations")
        print(f"   • category_analysis_{month_label}.png - Category-specific analysis")
        print(f"   • sales_report_{month_label}.txt - Complete report")

        return analyzer

    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        raise
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise


if __name__ == "__main__":
    # Interactive mode: ask user for year and month
    from datetime import datetime

    print("\n" + "=" * 60)
    print("SINGLE MONTH SALES ANALYZER")
    print("=" * 60)

    # Get current date as default
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.strftime("%B")  # Full month name (e.g., "October")

    print(f"\nCurrent date: {current_month} {current_year}")
    print("Press Enter to use current month or specify year and month.")

    # Ask for year
    year_input = input(f"\nYear [{current_year}]: ").strip()
    year = int(year_input) if year_input else current_year

    # Ask for month
    month_input = input(f"Month [{current_month}]: ").strip()
    month = month_input if month_input else current_month

    # Capitalize first letter if needed
    if month:
        month = month.capitalize()

    print(f"\n📊 Analyzing: {month} {year}")
    print("=" * 60 + "\n")

    # Run analysis
    analyzer = main(year=year, month=month)
