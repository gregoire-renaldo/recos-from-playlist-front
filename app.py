import streamlit as st
import pandas as pd
import requests
import os

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a beautiful, modern look
st.markdown("""
<style>
    /* Global Background & Font */
    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a1a2e;
        font-weight: 700;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background_clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 20px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
        color: #ffffff;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    
    /* Card Styling for Recommendations */
    .book-card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        border: 1px solid #e0e0e0;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .book-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }
    .book-title {
        color: #2c3e50;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .book-author {
        color: #7f8c8d;
        font-size: 0.95rem;
        margin-bottom: 10px;
        font-style: italic;
    }
    .book-desc {
        color: #34495e;
        font-size: 0.9rem;
        flex-grow: 1;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 4;
        -webkit-box-orient: vertical;
    }
    .similarity-badge {
        background-color: #4ECDC4;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        align-self: flex-start;
        margin-top: 15px;
    }
    
    /* Input Fields */
    .stMultiSelect, .stButton {
        max-width: 800px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
API_URL = "https://playlist-recommender-812256971571.europe-west1.run.app"
DATA_PATH = "../recos-from-playlist/package-docker/data/music_corpus.parquet"

# --- Helper Functions ---
@st.cache_data
def load_data():
    """Loads song data from parquet file."""
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at {DATA_PATH}. Please check the path.")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(DATA_PATH)
        # Create a display column for easier searching
        # Attempt to find relevant columns
        
        # Heuristics based on inspection
        title_col = 'track_name' if 'track_name' in df.columns else 'title_key' if 'title_key' in df.columns else df.columns[0]
        artist_col = 'track_artist' if 'track_artist' in df.columns else 'artist_key' if 'artist_key' in df.columns else None
        
        if artist_col:
             df["display_name"] = df[artist_col].astype(str) + " - " + df[title_col].astype(str)
        else:
             df["display_name"] = df[title_col].astype(str)
             
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# --- Main App ---

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4616/4616041.png", width=80)
    st.title("Model Selection")
    st.markdown("Choose your recommendation persona:")
    
    model_choice = st.radio(
        "Select Recommender:",
        ("Bertin", "Michelle"),
        index=0,
        help="Bertin uses deep learning (BERT) for contextual matching. Michelle uses numerical audio features."
    )
    
    if model_choice == "Bertin":
        st.info("**Bertin** 🤖\n\nExpert in context and meaning. Uses BERT embeddings to understand the 'vibe' of your playlist and match it to book descriptions.")
    else:
        st.info("**Michelle** 📊\n\nData-driven analyst. Uses raw audio features like tempo, energy, and valence to find books with similar emotional markers.")

# Main Content
st.title("🎵 Song to Book Recommender 📚")
st.markdown("### meaningful lists from your favorite tunes")

# Load Data
df_songs = load_data()

if not df_songs.empty:
    # Song Selection
    st.markdown("#### 1. Select Songs")
    selected_display_names = st.multiselect(
        "Search and add songs to your playlist:",
        options=df_songs["display_name"].unique(),
        placeholder="Type to search (e.g. 'Bohemian Rhapsody')..."
    )
    
    # Get IDs of selected songs
    if selected_display_names:
        selected_songs = df_songs[df_songs["display_name"].isin(selected_display_names)]
        # We need the index or ID depending on the model requirement.
        # Bertin uses 'playlist_ids' (indices from the dataframe probably, or IDs if they are indices)
        # Michelle uses 'song_ids'
        # Looking at API code:
        # Bertin: recommend_books_from_playlist_bert_idx(playlist_indices=payload.playlist_ids) -> uses songs_df.index
        # Michelle: recommend_books_from_song_ids(song_ids) -> uses index or positional int
        
        # So in both cases, passing the DataFrame Index seems correct/safest if the index matches the training data.
        # Let's use the Index values.
        input_ids = selected_songs.index.tolist()
        
        st.markdown("#### 2. Get Recommendations")
        if st.button("Recommend Books 🚀", type="primary"):
            with st.spinner(f"Asking {model_choice} for recommendations..."):
                try:
                    if model_choice == "Bertin":
                        endpoint = f"{API_URL}/recommend/bert"
                        payload = {"playlist_ids": input_ids, "top_k": 5}
                    else: # Michelle
                        endpoint = f"{API_URL}/recommend/numerical"
                        # Michelle might need one request per song or list of songs?
                        # API says: song_ids: List[int].
                        payload = {"song_ids": input_ids, "n_recommendations": 5}
                    
                    response = requests.post(endpoint, json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        
                        if results:
                            st.success(f"Found {len(results)} recommendations!")
                            
                            # Display Results
                            cols = st.columns(3)
                            for idx, book in enumerate(results):
                                with cols[idx % 3]:
                                    title = book.get('title') or book.get('book_title')
                                    author = book.get('author')
                                    desc = book.get('description', 'No description available.')
                                    score = book.get('similarity', 0) or book.get('similarity_score', 0)
                                    # Normalize score display
                                    score_pct = f"{score:.1%}" if score <= 1 else f"{score:.2f}"
                                    
                                    # Truncate desc if needed
                                    if len(desc) > 300:
                                        desc = desc[:300] + "..."

                                    st.markdown(f"""
                                    <div class="book-card">
                                        <div class="book-title">{title}</div>
                                        <div class="book-author">by {author}</div>
                                        <div class="book-desc">{desc}</div>
                                        <div class="similarity-badge">Match: {score_pct}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.warning("No recommendations found. Try different songs.")
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
else:
    st.info("Loading song database... if this takes too long, check the data path.")
