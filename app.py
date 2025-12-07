import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    /* Fixed H1 Visibility and Gradient */
    h1 {
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text; /* standard property */
        padding-bottom: 20px;
        line-height: 1.2; /* Ensure height for gradient */
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6;
        color: #333333;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] span {
        color: #333333 !important;
    }
    
    /* Card Styling */
    div.book-card-container {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        height: 100%;
        display: flex;
        flex-direction: column;
        transition: transform 0.2s;
    }
    div.book-card-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .book-title {
        color: #2c3e50;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 5px;
        height: 3.5rem; /* Fixed height for title alignment */
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .book-author {
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-bottom: 12px;
        font-style: italic;
    }
    .similarity-badge {
        display: inline-block;
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    
    /* Description text styling */
    .desc-text {
        color: #4a4a4a;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 10px;
    }

    /* Input Fields */
    .stMultiSelect, .stButton {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Expander customization */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        color: #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
API_URL = st.secrets.get("API_URL", "")
DATA_PATH = st.secrets.get("DATA_PATH", "songs_light.parquet")

# --- Helper Functions ---
@st.cache_data
def load_data():
    """Loads song data from parquet file."""
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at {DATA_PATH}. Please check the path.")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(DATA_PATH)
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

# --- Autosave/State Management ---
if "recommendations" not in st.session_state:
    st.session_state.recommendations = {"Bertin": None, "Michelle": None}
if "last_songs" not in st.session_state:
    st.session_state.last_songs = []

# --- Main App ---

# Sidebar
with st.sidebar:
    st.title("Model Selection")
    
    model_choice = st.radio(
        "Select Persona:",
        ("Bertin", "Michelle"),
        index=0
    )
    
    st.divider()
    
    if model_choice == "Bertin":
        st.image("https://api.dicebear.com/9.x/avataaars/svg?seed=Bertin&backgroundColor=b6e3f4", width=120)
        st.markdown("### Bertin 🤖")
        st.info("I analyze the **context and meaning** of your songs using BERT embeddings to find books with a matching vibe.")
    else:
        st.image("https://api.dicebear.com/9.x/avataaars/svg?seed=Michelle&backgroundColor=ffdfbf", width=120)
        st.markdown("### Michelle 📊")
        st.info("I look at the **numbers**—tempo, energy, valence. I'll find books that match the emotional curve of your playlist.")

# Main Content
st.title("🎵 Song to Book Recommender 📚")
st.markdown("### create your reading list from your playlist")

# Load Data
df_songs = load_data()

if not df_songs.empty:
    # Song Selection
    st.markdown("#### 1. Select Songs")
    selected_display_names = st.multiselect(
        "Search and add songs:",
        options=df_songs["display_name"].unique(),
        placeholder="Type to search (e.g. 'Bohemian Rhapsody')...",
        default=st.session_state.last_songs # restore selection if possible, though multiselect state is auto-handled usually
    )
    
    # Update last songs in state to track changes if needed, 
    # but Streamlit's widget state usually suffices. 
    # However, to know when to invalidate cache or warning users:
    
    # Get IDs of selected songs
    if selected_display_names:
        selected_songs = df_songs[df_songs["display_name"].isin(selected_display_names)]
        input_ids = selected_songs.index.tolist()
        
        st.markdown("#### 2. Get Recommendations")
        
        if st.button(f"Ask {model_choice} to Recommend 🚀", type="primary"):
            with st.spinner(f"{model_choice} is thinking..."):
                try:
                    if model_choice == "Bertin":
                        endpoint = f"{API_URL}/recommend/bert"
                        payload = {"playlist_ids": input_ids, "top_k": 6} # Get a few more for grid
                    else: # Michelle
                        endpoint = f"{API_URL}/recommend/numerical"
                        payload = {"song_ids": input_ids, "n_recommendations": 6}
                    
                    response = requests.post(endpoint, json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # Save to session state
                        st.session_state.recommendations[model_choice] = data.get("results", [])
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

        # --- Display Results ---
        # Always check if we have results for the CURRENT model in session state and display them
        current_results = st.session_state.recommendations.get(model_choice)
        
        if current_results:
            st.divider()
            st.markdown(f"### {model_choice}'s Picks for You")
            
            # Grid Layout
            cols = st.columns(3)
            for idx, book in enumerate(current_results):
                with cols[idx % 3]:
                    title = book.get('title') or book.get('book_title')
                    author = book.get('author')
                    desc = book.get('description', 'No description available.')
                    score = book.get('similarity', 0) or book.get('similarity_score', 0)
                    score_pct = f"{score:.0%}" if score <= 1 else f"{score:.2f}"
                    
                    # Layout: Container -> Badge -> Title -> Author -> Desc -> Read More
                    with st.container():
                        st.markdown(f"""
                        <div class="book-card-container">
                            <div class="similarity-badge">{score_pct} Match</div>
                            <div class="book-title" title="{title}">{title}</div>
                            <div class="book-author">by {author}</div>
                            <div style="flex-grow: 1;"></div> 
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Description logic using Expander for 'Read More'
                        # We show a truncated version (approx 5 lines) or just the expander
                        # User asked: "5 lines visible, and a button read more"
                        
                        # 5 lines is roughly 300-400 chars depending on width.
                        preview_length = 350
                        if len(desc) > preview_length:
                            preview = desc[:preview_length].rsplit(' ', 1)[0] + "..."
                            st.markdown(f"<div class='desc-text'>{preview}</div>", unsafe_allow_html=True)
                            with st.expander("Read more"):
                                st.markdown(desc)
                        else:
                             st.markdown(f"<div class='desc-text'>{desc}</div>", unsafe_allow_html=True)

        elif selected_display_names and not current_results:
             st.info(f"Click the button above to get recommendations from {model_choice}!")

else:
    st.info("Loading song database... if this takes too long, check the data path.")
