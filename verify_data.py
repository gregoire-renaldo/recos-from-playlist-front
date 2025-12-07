
import pandas as pd
import os

DATA_PATH = "../recos-from-playlist/package-docker/data/music_corpus.parquet"

def test_load():
    print(f"Checking {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print("File not found!")
        return

    try:
        df = pd.read_parquet(DATA_PATH)
        print("Columns found:", df.columns.tolist())
        
        title_col = 'track_name' if 'track_name' in df.columns else 'title_key' if 'title_key' in df.columns else df.columns[0]
        artist_col = 'track_artist' if 'track_artist' in df.columns else 'artist_key' if 'artist_key' in df.columns else None
        
        print(f"Selected Title Col: {title_col}")
        print(f"Selected Artist Col: {artist_col}")
        
        if artist_col:
             df["display_name"] = df[artist_col].astype(str) + " - " + df[title_col].astype(str)
        else:
             df["display_name"] = df[title_col].astype(str)
             
        print("Head of display name:")
        print(df["display_name"].head())
        print("SUCCESS")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_load()
