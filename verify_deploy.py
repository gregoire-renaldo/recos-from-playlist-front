import streamlit as st
import pandas as pd
import os
from app import load_data, DATA_PATH

def verify():
    print(f"configured DATA_PATH: {DATA_PATH}")
    if DATA_PATH != "songs_light.parquet":
        # It might still be reading from .env if load_dotenv() is active and .env has the old value
        # But for Cloud deployment (no .env), it should be songs_light.parquet.
        print("Note: DATA_PATH is NOT 'songs_light.parquet'. This might be due to your local .env file.")
    
    df = load_data()
    print(f"Data Loaded Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    if "display_name" in df.columns and not df.empty:
        print("SUCCESS: Data loaded and processed correctly.")
    else:
        print("FAILURE: Data not loaded correctly.")

if __name__ == "__main__":
    verify()
