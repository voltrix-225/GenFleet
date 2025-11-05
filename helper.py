"""
helper.py
Utility functions for file management and logging.
"""

import os

def ensure_dir(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📁 Created directory: {path}")
import re

def sanitize_columns(df):
    """Sanitize DataFrame column names to be XGBoost-safe."""
    df = df.copy()
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c).strip("_") for c in df.columns]
    return df
