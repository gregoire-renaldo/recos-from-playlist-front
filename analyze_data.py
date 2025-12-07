import pandas as pd
import os

DATA_PATH = "../recos-from-playlist/package-docker/data/music_corpus.parquet"
OUTPUT_PATH = "songs_light.parquet"

def analyze_size():
    if not os.path.exists(DATA_PATH):
        print(f"File not found at {DATA_PATH}")
        return

    df = pd.read_parquet(DATA_PATH)
    print(f"Original Shape: {df.shape}")
    print(f"Original Columns: {df.columns.tolist()}")
    
    # Identify relevant columns
    # Based on previous runs: likely 'track_name', 'track_artist' or similar keys.
    # Looking at heuristics in app.py: 
    # title_col = 'track_name' or 'title_key'
    # artist_col = 'track_artist' or 'artist_key'
    
    cols_to_keep = []
    
    if 'track_name' in df.columns:
        cols_to_keep.append('track_name')
    elif 'title_key' in df.columns:
        cols_to_keep.append('title_key')
        
    if 'track_artist' in df.columns:
        cols_to_keep.append('track_artist')
    elif 'artist_key' in df.columns:
        cols_to_keep.append('artist_key')
        
    # We also need the index or ID if it's not the index
    # The app uses df.index.tolist() as IDs. So we preserve the index.
    
    if not cols_to_keep:
        print("Could not identify display columns.")
        return

    print(f"Keeping columns: {cols_to_keep}")
    df_light = df[cols_to_keep].copy()
    
    # Save optimized file
    df_light.to_parquet(OUTPUT_PATH)
    
    original_size = os.path.getsize(DATA_PATH) / (1024 * 1024)
    new_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    
    print(f"Original Size: {original_size:.2f} MB")
    print(f"Optimized Size: {new_size:.2f} MB")
    print(f"Reduction: {(1 - new_size/original_size)*100:.1f}%")

if __name__ == "__main__":
    analyze_size()
