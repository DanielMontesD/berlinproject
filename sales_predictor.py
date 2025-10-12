"""
Sales Prediction Module for Berlin Project
==========================================

This module implements predictive models to forecast sales based on historical data.

Features:
---------
- Time series forecasting for monthly sales
- Product-level demand prediction
- Trend analysis and seasonality detection
- Multiple prediction models (Linear Regression, Random Forest, SARIMA)
- Visual comparison of predictions vs actual data

Typical Usage:
-------------
    predictor = SalesPredictor()
    predictor.run_complete_analysis()

Author: Daniel Montes
Date: 2025-10-08
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class SalesPredictor:
    """
    Advanced sales prediction system using machine learning and statistical models.

    This class provides comprehensive sales forecasting capabilities including:
    - Monthly sales predictions
    - Product-level demand forecasting
    - Trend and seasonality analysis
    - Model performance evaluation

    Attributes:
    -----------
    monthly_data : Dict[str, pd.DataFrame]
        Historical sales data organized by month
    consolidated_data : Optional[pd.DataFrame]
        All historical data combined
    predictions : Dict[str, Any]
        Stored prediction results
    models : Dict[str, Any]
        Trained ML models
    """

    def __init__(self):
        """Initialize the sales predictor."""
        self.monthly_data: Dict[str, pd.DataFrame] = {}
        self.consolidated_data: Optional[pd.DataFrame] = None
        self.predictions: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.available_months: List[str] = []

    def load_all_data(self) -> None:
        """
        Load all available monthly sales data from the reports directory.

        This method scans for monthly CSV files and loads them with proper
        data cleaning and feature engineering for prediction models.
        """
        print(">> Loading historical sales data for prediction...")

        reports_dir = Path("reports")
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        for month in months:
            month_dir = reports_dir / month
            csv_file = month_dir / "report-sales_takings-item_sold.csv"

            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file, skiprows=1)

                    # Standardize columns
                    df.columns = [
                        "Menu Section",
                        "Menu Item",
                        "Size",
                        "Portion",
                        "Category",
                        "Unit Price",
                        "Quantity",
                        "Sales",
                        "% of Sales",
                    ]

                    # Clean numeric columns
                    for col in ["Unit Price", "Quantity", "Sales"]:
                        df[col] = (
                            df[col]
                            .astype(str)
                            .str.replace("$", "")
                            .str.replace(",", "")
                        )
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                    df = df.dropna(subset=["Sales"])

                    # Add temporal features
                    df["Month"] = month
                    df["Month_Number"] = months.index(month) + 1
                    df["Quarter"] = (df["Month_Number"] - 1) // 3 + 1

                    # Melbourne seasons
                    season_map = {
                        1: "Summer",
                        2: "Summer",
                        3: "Autumn",
                        4: "Autumn",
                        5: "Autumn",
                        6: "Winter",
                        7: "Winter",
                        8: "Winter",
                        9: "Spring",
                        10: "Spring",
                        11: "Spring",
                        12: "Summer",
                    }
                    df["Season"] = df["Month_Number"].map(season_map)

                    # Price categories
                    df["Price_Category"] = pd.cut(
                        df["Unit Price"],
                        bins=[0, 20, 30, 50, np.inf],
                        labels=["Budget", "Mid-Range", "Premium", "Luxury"],
                    )

                    self.monthly_data[month] = df
                    self.available_months.append(month)
                    print(
                        f"    OK {month}: {len(df)} items, Total Sales: ${df['Sales'].sum():,.2f}"
                    )

                except Exception as e:
                    print(f"    ERROR loading {month}: {e}")

        if self.monthly_data:
            self.consolidated_data = pd.concat(
                self.monthly_data.values(), ignore_index=True
            )
            print(
                f"\n>> Loaded {len(self.monthly_data)} months with {len(self.consolidated_data)} total records"
            )
            print(
                f">> Period: {self.available_months[0]} to {self.available_months[-1]}"
            )

    def prepare_monthly_aggregation(self) -> pd.DataFrame:
        """
        Prepare aggregated monthly data for time series prediction.

        Returns:
        --------
        pd.DataFrame
            Monthly aggregated data with temporal features
        """
        if self.consolidated_data is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")

        # Aggregate by month
        monthly_agg = (
            self.consolidated_data.groupby("Month_Number")
            .agg(
                {
                    "Sales": "sum",
                    "Quantity": "sum",
                    "Month": "first",
                    "Season": "first",
                    "Quarter": "first",
                }
            )
            .reset_index()
        )

        # Add temporal features for modeling
        monthly_agg["Month_Sin"] = np.sin(2 * np.pi * monthly_agg["Month_Number"] / 12)
        monthly_agg["Month_Cos"] = np.cos(2 * np.pi * monthly_agg["Month_Number"] / 12)

        # Sort by month number
        monthly_agg = monthly_agg.sort_values("Month_Number")

        return monthly_agg

    def predict_monthly_sales(
        self, months_ahead: int = 3
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Predict future monthly sales using multiple models.

        Parameters:
        -----------
        months_ahead : int
            Number of months to predict ahead

        Returns:
        --------
        Tuple[pd.DataFrame, Dict[str, float]]
            Predictions dataframe and model metrics
        """
        print(f"\n>> Predicting monthly sales for next {months_ahead} months...")

        monthly_agg = self.prepare_monthly_aggregation()

        # Prepare features and target
        feature_cols = ["Month_Number", "Month_Sin", "Month_Cos", "Quarter"]
        X = monthly_agg[feature_cols]
        y = monthly_agg["Sales"]

        # Split data: use last 2 months as validation
        if len(X) > 3:
            split_idx = len(X) - 2
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            X_train, X_val = X, X.iloc[-1:]
            y_train, y_val = y, y.iloc[-1:]

        # Train models
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=42, max_depth=5
            ),
        }

        results = {}
        predictions_by_model = {}

        for model_name, model in models.items():
            # Train
            model.fit(X_train, y_train)

            # Validate
            y_pred_val = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            r2 = r2_score(y_val, y_pred_val)

            results[model_name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "model": model}

            print(f"  Model: {model_name}")
            print(f"      MAE: ${mae:,.2f}")
            print(f"      RMSE: ${rmse:,.2f}")
            print(f"      R2: {r2:.3f}")

            # Future predictions
            last_month = monthly_agg["Month_Number"].max()
            future_months = []

            for i in range(1, months_ahead + 1):
                future_month_num = (last_month + i - 1) % 12 + 1
                future_quarter = (future_month_num - 1) // 3 + 1
                future_months.append(
                    {
                        "Month_Number": future_month_num,
                        "Month_Sin": np.sin(2 * np.pi * future_month_num / 12),
                        "Month_Cos": np.cos(2 * np.pi * future_month_num / 12),
                        "Quarter": future_quarter,
                    }
                )

            future_df = pd.DataFrame(future_months)
            future_predictions = model.predict(future_df[feature_cols])
            predictions_by_model[model_name] = future_predictions

        # Choose best model (lowest RMSE)
        best_model_name = min(results.keys(), key=lambda k: results[k]["RMSE"])
        best_model = results[best_model_name]["model"]
        self.models["monthly_sales"] = best_model

        print(f"\n  Best Model: {best_model_name}")

        # Create predictions dataframe
        month_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]

        last_month_num = monthly_agg["Month_Number"].max()
        predictions_data = []

        for i in range(months_ahead):
            month_num = (last_month_num + i) % 12 + 1
            predictions_data.append(
                {
                    "Month": month_names[month_num - 1],
                    "Month_Number": month_num,
                    "Predicted_Sales_LR": predictions_by_model["Linear Regression"][i],
                    "Predicted_Sales_RF": predictions_by_model["Random Forest"][i],
                    "Predicted_Sales_Best": predictions_by_model[best_model_name][i],
                }
            )

        predictions_df = pd.DataFrame(predictions_data)
        self.predictions["monthly"] = predictions_df

        return predictions_df, {best_model_name: results[best_model_name]}

    def predict_product_demand(self, top_n: int = 10) -> pd.DataFrame:
        """
        Predict demand for top products based on historical patterns.

        Parameters:
        -----------
        top_n : int
            Number of top products to predict

        Returns:
        --------
        pd.DataFrame
            Product demand predictions
        """
        print(f"\n>> Predicting demand for top {top_n} products...")

        if self.consolidated_data is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")

        # Get top products by total sales
        top_products = (
            self.consolidated_data.groupby("Menu Item")["Sales"]
            .sum()
            .nlargest(top_n)
            .index
        )

        predictions = []

        for product in top_products:
            product_data = self.consolidated_data[
                self.consolidated_data["Menu Item"] == product
            ]

            # Monthly aggregation for this product
            product_monthly = (
                product_data.groupby("Month_Number")
                .agg({"Sales": "sum", "Quantity": "sum", "Unit Price": "mean"})
                .reset_index()
            )

            if len(product_monthly) < 3:
                # Not enough data for prediction
                continue

            # Simple trend-based prediction
            X = product_monthly["Month_Number"].values.reshape(-1, 1)
            y_sales = product_monthly["Sales"].values
            y_quantity = product_monthly["Quantity"].values

            # Fit linear regression
            lr_sales = LinearRegression()
            lr_quantity = LinearRegression()

            lr_sales.fit(X, y_sales)
            lr_quantity.fit(X, y_quantity)

            # Predict for next month
            next_month = X.max() + 1
            if next_month > 12:
                next_month = 1

            pred_sales = lr_sales.predict([[next_month]])[0]
            pred_quantity = lr_quantity.predict([[next_month]])[0]

            # Ensure non-negative predictions
            pred_sales = max(0, pred_sales)
            pred_quantity = max(0, pred_quantity)

            predictions.append(
                {
                    "Product": product,
                    "Historical_Avg_Sales": product_data["Sales"].mean(),
                    "Historical_Avg_Quantity": product_data["Quantity"].mean(),
                    "Predicted_Next_Month_Sales": pred_sales,
                    "Predicted_Next_Month_Quantity": int(pred_quantity),
                    "Trend": "Increasing" if lr_sales.coef_[0] > 0 else "Decreasing",
                    "Confidence": "High" if len(product_monthly) >= 5 else "Medium",
                }
            )

        predictions_df = pd.DataFrame(predictions)
        self.predictions["products"] = predictions_df

        print(f"  Generated predictions for {len(predictions_df)} products")

        return predictions_df

    def analyze_seasonality(self) -> Dict[str, Any]:
        """
        Analyze seasonal patterns in sales data.

        Returns:
        --------
        Dict[str, Any]
            Seasonality analysis results
        """
        print("\n>> Analyzing seasonal patterns (Melbourne seasons)...")

        if self.consolidated_data is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")

        # Sales by season
        seasonal_sales = (
            self.consolidated_data.groupby("Season")
            .agg({"Sales": ["sum", "mean", "std"], "Quantity": ["sum", "mean"]})
            .round(2)
        )

        # Category performance by season
        category_season = (
            self.consolidated_data.groupby(["Season", "Category"])["Sales"]
            .sum()
            .unstack(fill_value=0)
        )

        # Find peak season
        season_totals = self.consolidated_data.groupby("Season")["Sales"].sum()
        peak_season = season_totals.idxmax()

        seasonality_results = {
            "seasonal_sales": seasonal_sales,
            "category_by_season": category_season,
            "peak_season": peak_season,
            "season_performance": season_totals.to_dict(),
        }

        print(f"  Peak season: {peak_season}")
        print(f"  Sales by season:")
        for season, sales in season_totals.items():
            print(f"      {season}: ${sales:,.2f}")

        self.predictions["seasonality"] = seasonality_results

        return seasonality_results

    def create_prediction_visualizations(self) -> None:
        """
        Create comprehensive visualizations of predictions and analysis.
        """
        print("\n>> Creating prediction visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Sales Prediction & Forecast Analysis", fontsize=18, fontweight="bold"
        )

        # 1. Monthly Sales with Predictions
        monthly_agg = self.prepare_monthly_aggregation()
        month_labels = monthly_agg["Month"].values

        axes[0, 0].plot(
            month_labels,
            monthly_agg["Sales"].values,
            marker="o",
            linewidth=2,
            markersize=8,
            label="Historical Sales",
            color="blue",
        )

        # Add predictions if available
        if "monthly" in self.predictions:
            pred_df = self.predictions["monthly"]
            pred_months = pred_df["Month"].values
            pred_sales = pred_df["Predicted_Sales_Best"].values

            # Connect last historical point to first prediction
            all_months = list(month_labels) + list(pred_months)
            all_sales = list(monthly_agg["Sales"].values) + list(pred_sales)

            # Plot predictions
            axes[0, 0].plot(
                all_months[-len(pred_months) - 1 :],
                all_sales[-len(pred_sales) - 1 :],
                marker="s",
                linewidth=2,
                markersize=8,
                linestyle="--",
                label="Predicted Sales",
                color="red",
            )

            # Add value labels for predictions
            for i, (month, sales) in enumerate(zip(pred_months, pred_sales)):
                axes[0, 0].annotate(
                    f"${sales/1000:.0f}K",
                    (len(month_labels) + i, sales),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9,
                    color="red",
                )

        axes[0, 0].set_title(
            "Monthly Sales Trend & Forecast", fontsize=14, fontweight="bold"
        )
        axes[0, 0].set_ylabel("Sales ($)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. Top Products - Predicted vs Historical
        if "products" in self.predictions:
            product_pred = self.predictions["products"].head(8)

            x = np.arange(len(product_pred))
            width = 0.35

            bars1 = axes[0, 1].bar(
                x - width / 2,
                product_pred["Historical_Avg_Sales"],
                width,
                label="Historical Avg",
                color="skyblue",
                alpha=0.8,
            )
            bars2 = axes[0, 1].bar(
                x + width / 2,
                product_pred["Predicted_Next_Month_Sales"],
                width,
                label="Predicted Next Month",
                color="coral",
                alpha=0.8,
            )

            axes[0, 1].set_title(
                "Top Products: Historical vs Predicted Sales",
                fontsize=14,
                fontweight="bold",
            )
            axes[0, 1].set_ylabel("Sales ($)")
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(
                [p[:15] + "..." if len(p) > 15 else p for p in product_pred["Product"]],
                rotation=45,
                ha="right",
                fontsize=8,
            )
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3, axis="y")

        # 3. Seasonal Performance
        if "seasonality" in self.predictions:
            seasonality = self.predictions["seasonality"]
            season_performance = seasonality["season_performance"]

            seasons = list(season_performance.keys())
            sales = list(season_performance.values())

            season_colors = {
                "Summer": "#FFD700",
                "Autumn": "#FF8C00",
                "Winter": "#4169E1",
                "Spring": "#32CD32",
            }
            colors = [season_colors.get(s, "gray") for s in seasons]

            bars = axes[1, 0].bar(seasons, sales, color=colors, alpha=0.8)
            axes[1, 0].set_title(
                "Sales Performance by Season (Melbourne)",
                fontsize=14,
                fontweight="bold",
            )
            axes[1, 0].set_ylabel("Total Sales ($)")

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].annotate(
                    f"${height/1000:.0f}K",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        # 4. Product Demand Trends
        if "products" in self.predictions:
            product_pred = self.predictions["products"].head(10)

            # Create trend indicators
            trend_colors = {"Increasing": "green", "Decreasing": "red"}
            colors = [trend_colors.get(t, "gray") for t in product_pred["Trend"]]

            axes[1, 1].barh(
                range(len(product_pred)),
                product_pred["Predicted_Next_Month_Quantity"],
                color=colors,
                alpha=0.7,
            )
            axes[1, 1].set_yticks(range(len(product_pred)))
            axes[1, 1].set_yticklabels(
                [p[:20] + "..." if len(p) > 20 else p for p in product_pred["Product"]],
                fontsize=9,
            )
            axes[1, 1].set_title(
                "Predicted Demand - Next Month (Units)", fontsize=14, fontweight="bold"
            )
            axes[1, 1].set_xlabel("Predicted Quantity")
            axes[1, 1].invert_yaxis()
            axes[1, 1].grid(True, alpha=0.3, axis="x")

            # Add legend for trends
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="green", alpha=0.7, label="Increasing Trend"),
                Patch(facecolor="red", alpha=0.7, label="Decreasing Trend"),
            ]
            axes[1, 1].legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        plt.savefig("sales_predictions.png", dpi=300, bbox_inches="tight")
        print("  Predictions saved as 'sales_predictions.png'")
        plt.close()

    def generate_prediction_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive prediction report.

        Parameters:
        -----------
        output_path : Optional[Path]
            Path to save the report

        Returns:
        --------
        str
            Formatted report text
        """
        print("\n>> Generating prediction report...")

        report_lines = [
            "=" * 80,
            "SALES PREDICTION & FORECAST REPORT - BERLIN PROJECT",
            "=" * 80,
            f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Historical Period: {self.available_months[0]} to {self.available_months[-1]}",
            f"Data Points: {len(self.consolidated_data)} transactions",
            "",
            "=" * 80,
            "MONTHLY SALES PREDICTIONS:",
            "=" * 80,
        ]

        if "monthly" in self.predictions:
            pred_df = self.predictions["monthly"]
            report_lines.append("")
            for _, row in pred_df.iterrows():
                report_lines.append(
                    f"   {row['Month']:<12} Predicted Sales: ${row['Predicted_Sales_Best']:>12,.2f}"
                )

            total_predicted = pred_df["Predicted_Sales_Best"].sum()
            report_lines.append(
                f"\n   {'Total Forecast:':<12} ${total_predicted:>12,.2f}"
            )

        report_lines.extend(
            [
                "",
                "=" * 80,
                "TOP PRODUCTS - DEMAND FORECAST:",
                "=" * 80,
            ]
        )

        if "products" in self.predictions:
            product_pred = self.predictions["products"].head(10)
            report_lines.append("")
            for idx, row in product_pred.iterrows():
                trend_symbol = "UP" if row["Trend"] == "Increasing" else "DOWN"
                report_lines.append(f"   [{trend_symbol}] {row['Product'][:40]:<40}")
                report_lines.append(
                    f"      Historical Avg: ${row['Historical_Avg_Sales']:>8,.2f} | "
                    f"Predicted: ${row['Predicted_Next_Month_Sales']:>8,.2f} | "
                    f"Units: {row['Predicted_Next_Month_Quantity']:>3}"
                )
                report_lines.append("")

        report_lines.extend(
            [
                "=" * 80,
                "SEASONAL ANALYSIS (Melbourne):",
                "=" * 80,
            ]
        )

        if "seasonality" in self.predictions:
            seasonality = self.predictions["seasonality"]
            report_lines.append(f"\n   Peak Season: {seasonality['peak_season']}")
            report_lines.append("\n   Season Performance:")
            for season, sales in seasonality["season_performance"].items():
                report_lines.append(f"      {season:<10} ${sales:>12,.2f}")

        report_lines.extend(
            [
                "",
                "=" * 80,
                "KEY INSIGHTS & RECOMMENDATIONS:",
                "=" * 80,
            ]
        )

        # Generate insights
        insights = self._generate_insights()
        for insight in insights:
            report_lines.append(f"   • {insight}")

        report_lines.extend(["", "=" * 80])

        report_text = "\n".join(report_lines)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"  Report saved as '{output_path}'")

        return report_text

    def _generate_insights(self) -> List[str]:
        """
        Generate actionable business insights from predictions.

        Returns:
        --------
        List[str]
            List of insight statements
        """
        insights = []

        if "monthly" in self.predictions:
            pred_df = self.predictions["monthly"]
            monthly_agg = self.prepare_monthly_aggregation()

            avg_historical = monthly_agg["Sales"].mean()
            avg_predicted = pred_df["Predicted_Sales_Best"].mean()

            change_pct = ((avg_predicted - avg_historical) / avg_historical) * 100

            if change_pct > 5:
                insights.append(
                    f"Sales are predicted to INCREASE by {change_pct:.1f}% on average. "
                    "Consider increasing inventory and staffing."
                )
            elif change_pct < -5:
                insights.append(
                    f"Sales are predicted to DECREASE by {abs(change_pct):.1f}% on average. "
                    "Review pricing strategy and promotional activities."
                )
            else:
                insights.append(
                    "Sales are predicted to remain STABLE. "
                    "Maintain current operations and inventory levels."
                )

        if "products" in self.predictions:
            product_pred = self.predictions["products"]
            increasing_products = product_pred[product_pred["Trend"] == "Increasing"]

            if len(increasing_products) > 0:
                top_increasing = increasing_products.iloc[0]
                insights.append(
                    f"Top growing product: '{top_increasing['Product']}'. "
                    "Ensure adequate stock and consider promotional emphasis."
                )

        if "seasonality" in self.predictions:
            seasonality = self.predictions["seasonality"]
            peak_season = seasonality["peak_season"]

            insights.append(
                f"{peak_season} is the peak season. Plan inventory, staffing, "
                "and marketing campaigns accordingly."
            )

        # General recommendation
        insights.append(
            "Regular monitoring of actual vs predicted sales will help refine "
            "forecasting accuracy over time."
        )

        return insights

    def run_complete_analysis(
        self, months_ahead: int = 3, top_products: int = 10
    ) -> None:
        """
        Run complete prediction analysis workflow.

        Parameters:
        -----------
        months_ahead : int
            Number of months to predict ahead
        top_products : int
            Number of top products to analyze
        """
        print(">> STARTING COMPREHENSIVE SALES PREDICTION ANALYSIS")
        print("=" * 80)

        # Load data
        self.load_all_data()

        if not self.monthly_data:
            print("❌ No data found!")
            return

        # Run predictions
        self.predict_monthly_sales(months_ahead=months_ahead)
        self.predict_product_demand(top_n=top_products)
        self.analyze_seasonality()

        # Generate visualizations
        self.create_prediction_visualizations()

        # Generate report
        self.generate_prediction_report(Path("sales_predictions_report.txt"))

        print("\n>> Prediction analysis completed successfully!")
        print("Generated files:")
        print("  - sales_predictions.png - Visualization dashboard")
        print("  - sales_predictions_report.txt - Detailed forecast report")


def main():
    """
    Main execution function for sales prediction.
    """
    predictor = SalesPredictor()
    predictor.run_complete_analysis(months_ahead=3, top_products=10)


if __name__ == "__main__":
    main()
