"""
Shared Utility Module for Berlin Project
=======================================

This module contains common constants and helper functions used across
the sales analysis and prediction scripts.

Contents:
- Constants: MONTHS, SEASON_MAP, PRICE_BINS, PRICE_LABELS
- Functions: clean_currency, get_season, load_and_clean_sales_data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union

# --- Constants ---

MONTHS = [
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

# Melbourne seasons (Southern Hemisphere)
# Summer: Dec-Feb, Autumn: Mar-May, Winter: Jun-Aug, Spring: Sep-Nov
SEASON_MAP = {
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

PRICE_BINS = [0, 20, 30, 50, np.inf]
PRICE_LABELS = ["Budget", "Mid-Range", "Premium", "Luxury"]

# --- Helper Functions ---

def clean_currency(value: Union[str, float, int]) -> float:
    """
    Clean currency string and convert to float.
    Removes '$' and ',' characters.
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    if pd.isna(value):
        return 0.0
        
    return float(str(value).replace("$", "").replace(",", "").replace("%", ""))

def get_season(month_number: int) -> str:
    """Return the season for a given month number (1-12)."""
    return SEASON_MAP.get(month_number, "Unknown")

def load_and_clean_sales_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and clean sales data from a CSV file.
    
    Performs standard cleaning:
    - Skips first row (header)
    - Standardizes column names
    - Cleans numeric columns (Sales, Quantity, Unit Price)
    - Drops rows with missing sales
    
    Returns:
        pd.DataFrame: Cleaned data
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
        
    # Load CSV, skipping first row
    df = pd.read_csv(csv_path, skiprows=1)
    
    # Standardize column names
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
        # Use apply to handle mixed types safely
        df[col] = df[col].apply(clean_currency)
        
    # Clean percentage if needed, though usually not used for calculation
    if "% of Sales" in df.columns:
         df["% of Sales"] = df["% of Sales"].apply(clean_currency)

    # Remove invalid rows
    df = df.dropna(subset=["Sales"])
    
    return df
