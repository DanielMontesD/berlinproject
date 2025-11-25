"""
Advanced Sales Prediction Module for Berlin Project
===================================================

This module implements enhanced predictive models leveraging multi-year historical data
(2023-2025) to forecast sales with improved accuracy.

Enhanced Features:
------------------
- Multi-year data loading from restructured reports/ directory
- Advanced temporal features (year-over-year trends, multi-year seasonality)
- Multiple advanced ML models (XGBoost, Gradient Boosting, Random Forest)
- Cross-validation with proper time series splits
- Year-over-year comparison and trend analysis
- Enhanced visualizations with historical context

Key Improvements over basic predictor:
--------------------------------------
1. Uses 3 years of data instead of 1 year for better pattern recognition
2. Captures year-over-year growth trends
3. Detects multi-year seasonal patterns
4. Implements more sophisticated ML models
5. Provides confidence intervals for predictions

Typical Usage:
-------------
    predictor = AdvancedSalesPredictor()
    predictor.run_complete_analysis()

Author: Daniel Montes
Date: 2025-10-14
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from utils import (
    MONTHS,
    SEASON_MAP,
    PRICE_BINS,
    PRICE_LABELS,
    load_and_clean_sales_data,
)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class AdvancedSalesPredictor:
    """
    Advanced sales prediction system using multi-year historical data.

    This class provides enhanced forecasting capabilities by leveraging
    data from 2023, 2024, and 2025 to identify long-term trends and
    seasonal patterns.

    Attributes:
    -----------
    yearly_data : Dict[int, Dict[str, pd.DataFrame]]
        Historical sales data organized by year and month
    consolidated_data : Optional[pd.DataFrame]
        All historical data combined with temporal features
    time_series_data : Optional[pd.DataFrame]
        Monthly aggregated time series data
    predictions : Dict[str, Any]
        Stored prediction results
    models : Dict[str, Any]
        Trained ML models
    feature_importance : Dict[str, pd.DataFrame]
        Feature importance for each model
    """

    def __init__(self):
        """Initialize the advanced sales predictor."""
        self.yearly_data: Dict[int, Dict[str, pd.DataFrame]] = {}
        self.consolidated_data: Optional[pd.DataFrame] = None
        self.time_series_data: Optional[pd.DataFrame] = None
        self.predictions: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        self.scaler = StandardScaler()

    def load_multi_year_data(self, years: List[int] = [2023, 2024, 2025]) -> None:
        """
        Load sales data from multiple years with new directory structure.

        New structure: reports/YEAR/MONTH/report-sales_takings-item_sold.csv

        Parameters:
        -----------
        years : List[int]
            List of years to load data from
        """
        print(">> Loading multi-year historical sales data...")
        print(f">> Target years: {years}")

        reports_dir = Path("reports")
        reports_dir = Path("reports")
        months = MONTHS

        total_records = 0

        for year in years:
            year_dir = reports_dir / str(year)
            if not year_dir.exists():
                print(f"  ⚠️  Year {year} directory not found, skipping...")
                continue

            self.yearly_data[year] = {}
            year_records = 0

            for month_idx, month in enumerate(months, 1):
                month_dir = year_dir / month
                csv_file = month_dir / "report-sales_takings-item_sold.csv"

                if csv_file.exists():
                    try:
                        # Use shared utility to load and clean data
                        df = load_and_clean_sales_data(csv_file)

                        # Add temporal features
                        df["Year"] = year
                        df["Month"] = month
                        df["Month_Number"] = month_idx
                        df["Quarter"] = (month_idx - 1) // 3 + 1

                        # Create a datetime column for proper time series analysis
                        df["Date"] = pd.to_datetime(f"{year}-{month_idx:02d}-01")

                        # Add Melbourne seasons
                        df["Season"] = df["Month_Number"].map(SEASON_MAP)

                        # Price categories
                        df["Price_Category"] = pd.cut(
                            df["Unit Price"],
                            bins=PRICE_BINS,
                            labels=PRICE_LABELS,
                        )

                        self.yearly_data[year][month] = df
                        year_records += len(df)
                        print(
                            f"    OK {year}-{month}: {len(df)} items, ${df['Sales'].sum():,.2f}"
                        )

                    except Exception as e:
                        print(f"    XX ERROR loading {year}-{month}: {e}")

            if year_records > 0:
                total_records += year_records
                print(f"  >> {year} Total: {year_records} records\n")

        # Consolidate all data
        all_dataframes = []
        for year, months_data in self.yearly_data.items():
            all_dataframes.extend(months_data.values())

        if all_dataframes:
            self.consolidated_data = pd.concat(all_dataframes, ignore_index=True)
            self.consolidated_data = self.consolidated_data.sort_values("Date")

            print(f"\n>> 📊 TOTAL DATA LOADED:")
            print(f"   - Years: {len(self.yearly_data)}")
            print(f"   - Total Records: {total_records}")
            print(
                f"   - Date Range: {self.consolidated_data['Date'].min()} to {self.consolidated_data['Date'].max()}"
            )
            print(f"   - Total Sales: ${self.consolidated_data['Sales'].sum():,.2f}")
        else:
            print("\n>> ❌ No data loaded!")

    def prepare_time_series_features(self) -> pd.DataFrame:
        """
        Create enhanced time series dataset with advanced features.

        Features include:
        - Basic temporal: month, year, quarter
        - Cyclical encoding: sin/cos transformations
        - Lag features: previous month sales
        - Rolling statistics: moving averages, trends
        - Year-over-year comparisons
        - Trend features: linear time index

        Returns:
        --------
        pd.DataFrame
            Time series data with engineered features
        """
        if self.consolidated_data is None:
            raise ValueError("Data not loaded. Call load_multi_year_data() first.")

        print("\n>> Engineering advanced time series features...")

        # Aggregate by year-month
        monthly_agg = (
            self.consolidated_data.groupby(["Year", "Month_Number", "Date"])
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

        monthly_agg = monthly_agg.sort_values("Date").reset_index(drop=True)

        # Add time index (months since start)
        monthly_agg["Time_Index"] = range(len(monthly_agg))

        # Cyclical features for month (captures seasonality)
        monthly_agg["Month_Sin"] = np.sin(2 * np.pi * monthly_agg["Month_Number"] / 12)
        monthly_agg["Month_Cos"] = np.cos(2 * np.pi * monthly_agg["Month_Number"] / 12)

        # Lag features (previous months sales)
        for lag in [1, 2, 3]:
            monthly_agg[f"Sales_Lag_{lag}"] = monthly_agg["Sales"].shift(lag)
            monthly_agg[f"Quantity_Lag_{lag}"] = monthly_agg["Quantity"].shift(lag)

        # Rolling statistics (moving averages and trends)
        for window in [3, 6]:
            monthly_agg[f"Sales_MA_{window}"] = (
                monthly_agg["Sales"].rolling(window=window, min_periods=1).mean()
            )
            monthly_agg[f"Sales_Std_{window}"] = (
                monthly_agg["Sales"].rolling(window=window, min_periods=1).std()
            )

        # Year-over-year growth
        monthly_agg["YoY_Growth"] = (
            monthly_agg.groupby("Month_Number")["Sales"].pct_change(periods=12) * 100
        )

        # Trend: difference from moving average
        monthly_agg["Trend_Deviation"] = (
            monthly_agg["Sales"] - monthly_agg["Sales_MA_6"]
        )

        # Season encoding (one-hot)
        season_dummies = pd.get_dummies(monthly_agg["Season"], prefix="Season")
        monthly_agg = pd.concat([monthly_agg, season_dummies], axis=1)

        # Average price per item
        monthly_agg["Avg_Price"] = monthly_agg["Sales"] / monthly_agg[
            "Quantity"
        ].replace(0, 1)

        print(f"   OK Created {len(monthly_agg)} monthly time series records")
        print(f"   OK Generated {len(monthly_agg.columns)} features")

        self.time_series_data = monthly_agg
        return monthly_agg

    def train_advanced_models(self, forecast_months: int = 3) -> Dict[str, Any]:
        """
        Train multiple advanced ML models with time series cross-validation.

        Models:
        -------
        1. XGBoost Regressor (gradient boosting)
        2. Random Forest Regressor (ensemble)
        3. Gradient Boosting Regressor (boosting)

        Parameters:
        -----------
        forecast_months : int
            Number of months to forecast

        Returns:
        --------
        Dict[str, Any]
            Dictionary with model results and metrics
        """
        if self.time_series_data is None:
            self.prepare_time_series_features()

        print(
            f"\n>> Training advanced ML models (forecasting {forecast_months} months)..."
        )

        # Select features for modeling
        feature_cols = [
            "Time_Index",
            "Year",
            "Month_Number",
            "Quarter",
            "Month_Sin",
            "Month_Cos",
            "Sales_Lag_1",
            "Sales_Lag_2",
            "Sales_Lag_3",
            "Sales_MA_3",
            "Sales_MA_6",
            "Avg_Price",
        ]

        # Add season dummies if they exist
        season_cols = [
            col for col in self.time_series_data.columns if col.startswith("Season_")
        ]
        feature_cols.extend(season_cols)

        # Store feature columns for later use in predictions
        self.feature_cols = feature_cols

        # Remove rows with NaN from lag/rolling features
        data_clean = self.time_series_data.dropna(subset=feature_cols)

        if len(data_clean) < 10:
            print("   ⚠️  Insufficient data for training. Need at least 10 months.")
            return {}

        X = data_clean[feature_cols]
        y = data_clean["Sales"]

        # Time series split for validation
        n_splits = min(2, max(1, len(X) // 10))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Define models with optimized hyperparameters for small datasets
        models = {
            "XGBoost": xgb.XGBRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                objective="reg:squarederror",
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                min_samples_split=5,
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            ),
        }

        results = {}

        for model_name, model in models.items():
            print(f"\n   >> Training {model_name}...")

            # Cross-validation
            cv_scores = {"MAE": [], "RMSE": [], "R2": [], "MAPE": []}

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                cv_scores["MAE"].append(mean_absolute_error(y_val, y_pred))
                cv_scores["RMSE"].append(np.sqrt(mean_squared_error(y_val, y_pred)))
                cv_scores["R2"].append(r2_score(y_val, y_pred))
                cv_scores["MAPE"].append(
                    mean_absolute_percentage_error(y_val, y_pred) * 100
                )

            # Train on all data for final model
            model.fit(X, y)

            # Calculate average metrics
            avg_metrics = {
                "MAE": np.mean(cv_scores["MAE"]),
                "RMSE": np.mean(cv_scores["RMSE"]),
                "R2": np.mean(cv_scores["R2"]),
                "MAPE": np.mean(cv_scores["MAPE"]),
                "model": model,
            }

            results[model_name] = avg_metrics

            print(f"      MAE:  ${avg_metrics['MAE']:,.2f}")
            print(f"      RMSE: ${avg_metrics['RMSE']:,.2f}")
            print(f"      R²:   {avg_metrics['R2']:.3f}")
            print(f"      MAPE: {avg_metrics['MAPE']:.2f}%")

            # Feature importance for tree-based models
            if hasattr(model, "feature_importances_"):
                importance_df = pd.DataFrame(
                    {"Feature": feature_cols, "Importance": model.feature_importances_}
                ).sort_values("Importance", ascending=False)

                self.feature_importance[model_name] = importance_df
                print(
                    f"      Top 3 Features: {', '.join(importance_df.head(3)['Feature'].tolist())}"
                )

        # Select best model based on RMSE
        best_model_name = min(results.keys(), key=lambda k: results[k]["RMSE"])
        print(f"\n   >> Best Model: {best_model_name}")
        print(f"      (RMSE: ${results[best_model_name]['RMSE']:,.2f})")

        self.models["best_model"] = results[best_model_name]["model"]
        self.models["best_model_name"] = best_model_name
        self.models["all_results"] = results

        return results

    def predict_future_sales(self, months_ahead: int = 3) -> pd.DataFrame:
        """
        Generate predictions for future months using trained models.

        This method uses iterative prediction where each month's prediction
        is used to calculate lag features and moving averages for subsequent
        months, allowing for dynamic forecasting that adapts to predicted trends.

        Parameters:
        -----------
        months_ahead : int
            Number of months to predict into the future

        Returns:
        --------
        pd.DataFrame
            Predictions with confidence intervals
        """
        print(f"\n>> Generating predictions for next {months_ahead} months...")

        if "best_model" not in self.models:
            print("   ⚠️  No trained model found. Training models first...")
            self.train_advanced_models()

        if self.time_series_data is None:
            raise ValueError("Time series data not prepared.")

        # Get the last known data point
        last_data = self.time_series_data.iloc[-1]
        last_date = last_data["Date"]
        last_time_idx = last_data["Time_Index"]

        # Generate future data points
        predictions_list = []

        # Store all sales (historical + predictions) for dynamic lag calculation
        all_sales = list(self.time_series_data["Sales"].values)
        all_quantities = list(self.time_series_data["Quantity"].values)

        for i in range(1, months_ahead + 1):
            future_date = last_date + pd.DateOffset(months=i)
            future_month = future_date.month
            future_year = future_date.year
            future_quarter = (future_month - 1) // 3 + 1

            # Create feature dictionary
            future_features = {
                "Time_Index": last_time_idx + i,
                "Year": future_year,
                "Month_Number": future_month,
                "Quarter": future_quarter,
                "Month_Sin": np.sin(2 * np.pi * future_month / 12),
                "Month_Cos": np.cos(2 * np.pi * future_month / 12),
            }

            # Add lag features (use historical + previous predictions)
            for lag in [1, 2, 3]:
                # Calculate the index in our combined historical + predictions array
                sales_idx = len(self.time_series_data) + i - 1 - lag

                if sales_idx >= 0 and sales_idx < len(all_sales):
                    future_features[f"Sales_Lag_{lag}"] = all_sales[sales_idx]
                else:
                    # Fallback to mean if not enough history
                    future_features[f"Sales_Lag_{lag}"] = self.time_series_data[
                        "Sales"
                    ].mean()

                # Similar for quantity lags
                if sales_idx >= 0 and sales_idx < len(all_quantities):
                    future_features[f"Quantity_Lag_{lag}"] = all_quantities[sales_idx]
                else:
                    future_features[f"Quantity_Lag_{lag}"] = self.time_series_data[
                        "Quantity"
                    ].mean()

            # Add moving averages (use recent data + predictions)
            for window in [3, 6]:
                # Get the last 'window' values including predictions
                start_idx = len(self.time_series_data) + i - window
                end_idx = len(self.time_series_data) + i

                if start_idx >= 0:
                    ma_values = all_sales[start_idx:end_idx]
                    future_features[f"Sales_MA_{window}"] = (
                        np.mean(ma_values)
                        if ma_values
                        else self.time_series_data["Sales"].mean()
                    )
                else:
                    # Mix historical and predictions
                    historical_needed = abs(start_idx)
                    historical_values = list(
                        self.time_series_data["Sales"].iloc[-historical_needed:].values
                    )
                    prediction_values = all_sales[len(self.time_series_data) : end_idx]
                    ma_values = historical_values + prediction_values
                    future_features[f"Sales_MA_{window}"] = (
                        np.mean(ma_values)
                        if ma_values
                        else self.time_series_data["Sales"].mean()
                    )

            # Add average price (using historical average as proxy)
            future_features["Avg_Price"] = self.time_series_data.iloc[-6:][
                "Avg_Price"
            ].mean()

            # Add season dummies
            future_season = SEASON_MAP[future_month]

            for season in ["Summer", "Autumn", "Winter", "Spring"]:
                future_features[f"Season_{season}"] = (
                    1 if season == future_season else 0
                )

            # Create DataFrame with features
            future_df = pd.DataFrame([future_features])

            # Ensure all required columns exist with correct order
            for col in self.feature_cols:
                if col not in future_df.columns:
                    future_df[col] = 0

            # Reorder columns to match training data
            future_df = future_df[self.feature_cols]

            # Make predictions with all models
            predictions = {}
            for model_name, model_info in self.models.get("all_results", {}).items():
                model = model_info["model"]
                pred = model.predict(future_df)[0]
                predictions[f"Pred_{model_name.replace(' ', '_')}"] = pred

            # Best model prediction
            best_pred = self.models["best_model"].predict(future_df)[0]

            # Add this prediction to our running list for future lag calculations
            all_sales.append(best_pred)
            # Estimate quantity based on average price
            estimated_qty = (
                best_pred / future_features["Avg_Price"]
                if future_features["Avg_Price"] > 0
                else 0
            )
            all_quantities.append(estimated_qty)

            # Calculate confidence interval (±15% as approximation)
            ci_lower = best_pred * 0.85
            ci_upper = best_pred * 1.15

            month_names = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]

            predictions_list.append(
                {
                    "Date": future_date,
                    "Month": month_names[future_month - 1],
                    "Year": future_year,
                    "Predicted_Sales": best_pred,
                    "CI_Lower": ci_lower,
                    "CI_Upper": ci_upper,
                    **predictions,
                }
            )

            print(
                f"   {future_year}-{month_names[future_month-1]}: ${best_pred:,.2f} (±{(ci_upper-ci_lower)/2:,.0f})"
            )

        predictions_df = pd.DataFrame(predictions_list)
        self.predictions["future_sales"] = predictions_df

        return predictions_df

    def analyze_year_over_year_trends(self) -> Dict[str, Any]:
        """
        Analyze year-over-year trends and growth patterns.

        Returns:
        --------
        Dict[str, Any]
            Year-over-year analysis results
        """
        print("\n>> Analyzing year-over-year trends...")

        if self.consolidated_data is None:
            raise ValueError("Data not loaded.")

        # Yearly totals
        yearly_sales = self.consolidated_data.groupby("Year")["Sales"].sum()
        yearly_qty = self.consolidated_data.groupby("Year")["Quantity"].sum()

        # Year-over-year growth
        yoy_growth = yearly_sales.pct_change() * 100

        # Monthly comparison across years
        monthly_by_year = self.consolidated_data.pivot_table(
            values="Sales", index="Month_Number", columns="Year", aggfunc="sum"
        )

        # Average monthly sales by year
        avg_monthly_by_year = monthly_by_year.mean()

        results = {
            "yearly_sales": yearly_sales.to_dict(),
            "yearly_quantity": yearly_qty.to_dict(),
            "yoy_growth": yoy_growth.to_dict(),
            "monthly_comparison": monthly_by_year,
            "avg_monthly": avg_monthly_by_year.to_dict(),
        }

        print("   Year-over-Year Sales:")
        for year, sales in yearly_sales.items():
            growth = yoy_growth.get(year, 0)
            print(f"      {year}: ${sales:,.2f} ({growth:+.1f}% YoY)")

        self.predictions["yoy_analysis"] = results
        return results

    def create_advanced_visualizations(self) -> None:
        """
        Create comprehensive visualizations with multi-year context.

        Generates 5 key visualizations in optimized layout:
        - Top (full width): Sales Forecast with Historical Context
        - Middle row: Monthly Sales Pattern | Seasonal Sales by Year
        - Bottom row: Year-over-Year Comparison | Model Performance (RMSE)
        """
        print("\n>> Creating advanced prediction visualizations...")

        fig = plt.figure(figsize=(20, 14))
        # Layout: 1 arriba (full width), luego 2x2 grid
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.25, height_ratios=[1, 1, 1])

        fig.suptitle(
            "Advanced Sales Forecasting - Multi-Year Analysis (2023-2025)",
            fontsize=20,
            fontweight="bold",
            y=0.98,
        )

        # 1. Historical Sales + Predictions (Top - Full Width)
        ax1 = fig.add_subplot(gs[0, :])

        if self.time_series_data is not None:
            historical = self.time_series_data.copy()

            # Plot historical data
            ax1.plot(
                historical["Date"],
                historical["Sales"],
                marker="o",
                linewidth=2,
                markersize=6,
                label="Historical Sales",
                color="#2E86AB",
            )

            # Add predictions if available
            if "future_sales" in self.predictions:
                pred_df = self.predictions["future_sales"]

                # Connect last historical to first prediction
                connection_dates = [historical["Date"].iloc[-1]] + list(pred_df["Date"])
                connection_sales = [historical["Sales"].iloc[-1]] + list(
                    pred_df["Predicted_Sales"]
                )

                ax1.plot(
                    connection_dates,
                    connection_sales,
                    marker="s",
                    linewidth=2,
                    markersize=7,
                    linestyle="--",
                    label="Predicted Sales",
                    color="#E63946",
                )

                # Confidence interval
                ax1.fill_between(
                    pred_df["Date"],
                    pred_df["CI_Lower"],
                    pred_df["CI_Upper"],
                    alpha=0.2,
                    color="#E63946",
                    label="Confidence Interval",
                )

            ax1.set_title(
                "Sales Forecast with Historical Context", fontsize=14, fontweight="bold"
            )
            ax1.set_ylabel("Sales ($)", fontsize=11)
            ax1.legend(loc="best")
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis="x", rotation=45)

        # 2. Monthly Pattern Across Years (Middle Left)
        ax2 = fig.add_subplot(gs[1, 0])

        if "yoy_analysis" in self.predictions:
            monthly_comparison = self.predictions["yoy_analysis"]["monthly_comparison"]

            month_names = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]

            for year in monthly_comparison.columns:
                ax2.plot(
                    range(1, 13),
                    monthly_comparison[year],
                    marker="o",
                    linewidth=2,
                    label=f"{year}",
                    markersize=6,
                )

            ax2.set_title(
                "Monthly Sales Pattern - Multi-Year Comparison",
                fontsize=14,
                fontweight="bold",
            )
            ax2.set_ylabel("Sales ($)", fontsize=11)
            ax2.set_xlabel("Month", fontsize=11)
            ax2.set_xticks(range(1, 13))
            ax2.set_xticklabels(month_names)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. Seasonal Pattern Heatmap (Middle Right)
        ax3 = fig.add_subplot(gs[1, 1])

        if self.consolidated_data is not None:
            seasonal_pivot = self.consolidated_data.pivot_table(
                values="Sales", index="Season", columns="Year", aggfunc="sum"
            )

            # Reorder seasons
            season_order = ["Summer", "Autumn", "Winter", "Spring"]
            seasonal_pivot = seasonal_pivot.reindex(season_order)

            sns.heatmap(
                seasonal_pivot,
                annot=True,
                fmt=".0f",
                cmap="YlOrRd",
                ax=ax3,
                cbar_kws={"label": "Sales ($)"},
            )

            ax3.set_title(
                "Seasonal Sales by Year (Melbourne)", fontsize=14, fontweight="bold"
            )
            ax3.set_xlabel("Year", fontsize=11)
            ax3.set_ylabel("Season", fontsize=11)

        # 4. Year-over-Year Comparison (Bottom Left)
        ax4 = fig.add_subplot(gs[2, 0])

        if "yoy_analysis" in self.predictions:
            yoy = self.predictions["yoy_analysis"]
            yearly_sales = yoy["yearly_sales"]

            years = list(yearly_sales.keys())
            sales = list(yearly_sales.values())
            colors = ["#457B9D", "#2A9D8F", "#E76F51"][: len(years)]

            bars = ax4.bar(years, sales, color=colors, alpha=0.8, width=0.6)

            # Add value labels
            for bar, sale in zip(bars, sales):
                height = bar.get_height()
                ax4.annotate(
                    f"${sale/1000:.0f}K",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontweight="bold",
                    fontsize=10,
                )

            # Add growth arrows
            for i in range(1, len(sales)):
                growth = (sales[i] - sales[i - 1]) / sales[i - 1] * 100
                mid_x = years[i - 1] + 0.5
                mid_y = (sales[i] + sales[i - 1]) / 2

                arrow_props = dict(
                    arrowstyle="->", color="green" if growth > 0 else "red", lw=2
                )

                ax4.annotate(
                    f"{growth:+.1f}%",
                    xy=(mid_x, mid_y),
                    fontsize=10,
                    fontweight="bold",
                    color="green" if growth > 0 else "red",
                    ha="center",
                )

            ax4.set_title(
                "Year-over-Year Sales Comparison", fontsize=14, fontweight="bold"
            )
            ax4.set_ylabel("Total Sales ($)", fontsize=11)
            ax4.set_xlabel("Year", fontsize=11)
            ax4.grid(True, alpha=0.3, axis="y")

        # 5. Model Performance Comparison (Bottom Right)
        ax5 = fig.add_subplot(gs[2, 1])

        if "all_results" in self.models:
            results = self.models["all_results"]
            model_names = list(results.keys())
            rmse_values = [results[m]["RMSE"] for m in model_names]

            colors_map = {
                "XGBoost": "#FF6B6B",
                "Random Forest": "#4ECDC4",
                "Gradient Boosting": "#45B7D1",
                "Ridge Regression": "#96CEB4",
            }
            colors = [colors_map.get(m, "gray") for m in model_names]

            bars = ax5.barh(model_names, rmse_values, color=colors, alpha=0.8)

            # Highlight best model
            best_idx = rmse_values.index(min(rmse_values))
            bars[best_idx].set_edgecolor("gold")
            bars[best_idx].set_linewidth(3)

            # Add value labels
            for bar, rmse in zip(bars, rmse_values):
                width = bar.get_width()
                ax5.annotate(
                    f"${rmse:,.0f}",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=10,
                )

            ax5.set_title("Model Performance (RMSE)", fontsize=14, fontweight="bold")
            ax5.set_xlabel("RMSE ($)", fontsize=11)
            ax5.invert_yaxis()
            ax5.grid(True, alpha=0.3, axis="x")

        plt.savefig("multi_year_forecast.png", dpi=300, bbox_inches="tight")
        print("   OK Saved as 'multi_year_forecast.png'")
        plt.show()  # Muestra la imagen automáticamente al usuario
        plt.close()

    def generate_advanced_report(
        self, output_path: str = "multi_year_forecast_report.txt"
    ) -> str:
        """
        Generate comprehensive prediction report with multi-year insights.

        Parameters:
        -----------
        output_path : str
            Path to save the report

        Returns:
        --------
        str
            Formatted report text
        """
        print("\n>> Generating advanced prediction report...")

        report_lines = [
            "=" * 90,
            "ADVANCED SALES FORECASTING REPORT - BERLIN PROJECT",
            "Multi-Year Analysis & Machine Learning Predictions",
            "=" * 90,
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model Used: {self.models.get('best_model_name', 'N/A')}",
            "",
            "=" * 90,
            "DATA SUMMARY:",
            "=" * 90,
        ]

        if self.consolidated_data is not None:
            date_range = f"{self.consolidated_data['Date'].min().strftime('%Y-%m')} to {self.consolidated_data['Date'].max().strftime('%Y-%m')}"
            total_sales = self.consolidated_data["Sales"].sum()
            total_records = len(self.consolidated_data)

            report_lines.extend(
                [
                    f"   Historical Period: {date_range}",
                    f"   Total Transactions: {total_records:,}",
                    f"   Total Sales: ${total_sales:,.2f}",
                    f"   Years Analyzed: {sorted(self.consolidated_data['Year'].unique())}",
                    "",
                ]
            )

        # Year-over-Year Analysis
        if "yoy_analysis" in self.predictions:
            report_lines.extend(["=" * 90, "YEAR-OVER-YEAR PERFORMANCE:", "=" * 90, ""])

            yoy = self.predictions["yoy_analysis"]
            for year, sales in yoy["yearly_sales"].items():
                growth = yoy["yoy_growth"].get(year, 0)
                avg_monthly = yoy["avg_monthly"].get(year, 0)

                growth_str = (
                    f"({growth:+.1f}% YoY)" if not np.isnan(growth) else "(baseline)"
                )
                report_lines.append(
                    f"   {year}: ${sales:>15,.2f} {growth_str:>15} | Avg Monthly: ${avg_monthly:,.2f}"
                )

            report_lines.append("")

        # Model Performance
        if "all_results" in self.models:
            report_lines.extend(["=" * 90, "MODEL PERFORMANCE METRICS:", "=" * 90, ""])

            for model_name, metrics in self.models["all_results"].items():
                is_best = model_name == self.models.get("best_model_name")
                marker = "🏆" if is_best else "  "

                report_lines.extend(
                    [
                        f"{marker} {model_name}:",
                        f"      MAE:  ${metrics['MAE']:>12,.2f}",
                        f"      RMSE: ${metrics['RMSE']:>12,.2f}",
                        f"      R²:   {metrics['R2']:>12.3f}",
                        f"      MAPE: {metrics['MAPE']:>11.2f}%",
                        "",
                    ]
                )

        # Future Predictions
        if "future_sales" in self.predictions:
            report_lines.extend(
                ["=" * 90, "SALES FORECAST - NEXT MONTHS:", "=" * 90, ""]
            )

            pred_df = self.predictions["future_sales"]
            total_forecast = pred_df["Predicted_Sales"].sum()

            for _, row in pred_df.iterrows():
                report_lines.append(
                    f"   {row['Year']}-{row['Month']:<3}: ${row['Predicted_Sales']:>12,.2f} "
                    f"(Range: ${row['CI_Lower']:,.0f} - ${row['CI_Upper']:,.0f})"
                )

            report_lines.extend(
                ["", f"   {'Total Forecast:':<8} ${total_forecast:>12,.2f}", ""]
            )

        # Key Insights
        report_lines.extend(["=" * 90, "KEY INSIGHTS & RECOMMENDATIONS:", "=" * 90, ""])

        insights = self._generate_advanced_insights()
        for insight in insights:
            report_lines.append(f"   • {insight}")

        report_lines.extend(["", "=" * 90])

        report_text = "\n".join(report_lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(f"   OK Report saved as '{output_path}'")

        return report_text

    def _generate_advanced_insights(self) -> List[str]:
        """
        Generate actionable business insights from multi-year analysis.

        Returns:
        --------
        List[str]
            List of insight statements
        """
        insights = []

        # Year-over-year growth insights
        if "yoy_analysis" in self.predictions:
            yoy = self.predictions["yoy_analysis"]
            years = sorted(yoy["yearly_sales"].keys())

            if len(years) >= 2:
                recent_growth = yoy["yoy_growth"].get(years[-1], 0)

                if recent_growth > 5:
                    insights.append(
                        f"Strong growth trajectory: Sales increased {recent_growth:.1f}% in {years[-1]}. "
                        "Consider expanding inventory and marketing efforts."
                    )
                elif recent_growth < -5:
                    insights.append(
                        f"Sales declined {abs(recent_growth):.1f}% in {years[-1]}. "
                        "Review market conditions, pricing, and customer feedback."
                    )
                else:
                    insights.append(
                        f"Stable performance: Sales growth at {recent_growth:.1f}% in {years[-1]}. "
                        "Focus on operational efficiency and customer retention."
                    )

        # Seasonal insights
        if self.consolidated_data is not None:
            seasonal_avg = self.consolidated_data.groupby("Season")["Sales"].mean()
            peak_season = seasonal_avg.idxmax()
            low_season = seasonal_avg.idxmin()

            insights.append(
                f"Seasonal pattern identified: {peak_season} is peak season, {low_season} is slowest. "
                f"Plan inventory and staffing accordingly."
            )

        # Prediction confidence
        if "future_sales" in self.predictions and "all_results" in self.models:
            best_model = self.models["best_model_name"]
            mape = self.models["all_results"][best_model]["MAPE"]

            if mape < 10:
                confidence = "High"
            elif mape < 20:
                confidence = "Moderate"
            else:
                confidence = "Low"

            insights.append(
                f"Forecast confidence: {confidence} (MAPE: {mape:.1f}%). "
                f"Model: {best_model} provides best accuracy."
            )

        # Long-term trend
        if self.time_series_data is not None and len(self.time_series_data) >= 12:
            recent_trend = self.time_series_data.iloc[-6:]["Sales"].mean()
            older_trend = self.time_series_data.iloc[-12:-6]["Sales"].mean()
            trend_change = (recent_trend - older_trend) / older_trend * 100

            if abs(trend_change) > 5:
                direction = "upward" if trend_change > 0 else "downward"
                insights.append(
                    f"Recent {direction} trend detected: Last 6 months avg ${recent_trend:,.0f} "
                    f"vs previous 6 months ${older_trend:,.0f} ({trend_change:+.1f}%)."
                )

        insights.append(
            "Continuous monitoring recommended: Update forecasts monthly with new data "
            "to maintain prediction accuracy."
        )

        return insights

    def run_complete_analysis(
        self, years: List[int] = [2023, 2024, 2025], forecast_months: int = 3
    ) -> None:
        """
        Run complete advanced prediction analysis workflow.

        Parameters:
        -----------
        years : List[int]
            Years to load data from
        forecast_months : int
            Number of months to forecast
        """
        print("\n" + "=" * 90)
        print(">> ADVANCED SALES PREDICTION SYSTEM - MULTI-YEAR ANALYSIS")
        print("=" * 90 + "\n")

        # 1. Load multi-year data
        self.load_multi_year_data(years=years)

        if not self.yearly_data:
            print("\n❌ No data found! Check your reports directory structure.")
            return

        # 2. Prepare time series features
        self.prepare_time_series_features()

        # 3. Train advanced models
        self.train_advanced_models(forecast_months=forecast_months)

        # 4. Generate predictions
        self.predict_future_sales(months_ahead=forecast_months)

        # 5. Analyze year-over-year trends
        self.analyze_year_over_year_trends()

        # 6. Create visualizations
        self.create_advanced_visualizations()

        # 7. Generate report
        self.generate_advanced_report()

        print("\n" + "=" * 90)
        print("OK ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 90)
        print("\nGenerated Files:")
        print("   >> multi_year_forecast.png - Multi-year visualization dashboard")
        print("   >> multi_year_forecast_report.txt - Comprehensive forecast report")
        print("\nNext Steps:")
        print("   1. Review predictions and confidence intervals")
        print("   2. Compare with actual sales as they occur")
        print("   3. Re-run monthly to improve accuracy")
        print("=" * 90 + "\n")


def main():
    """
    Main execution function for advanced sales prediction.
    """
    predictor = AdvancedSalesPredictor()
    predictor.run_complete_analysis(years=[2023, 2024, 2025], forecast_months=3)


if __name__ == "__main__":
    main()
