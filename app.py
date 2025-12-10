import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# Load environment variables
load_dotenv()

# --- Configuration & Styling - --
st.set_page_config(
    page_title="Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css(file_name: str = "styles.css"):
    """Inject custom CSS from a local file."""
    if os.path.exists(file_name):
        with open(file_name) as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file '{file_name}' not found.")

# Load external CSS
load_css("styles.css")

# --- Constants ---
# don't forget to change it between dev and prod
API_URL = st.secrets.get("API_URL", "")
# API_URL = "http://0.0.0.0:8000"
DATA_PATH = st.secrets.get("DATA_PATH", "song_corpus_sorted_light.parquet")
GOOGLE_BOOKS_API_KEY = st.secrets.get("GOOGLE_BOOKS_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
if not GOOGLE_BOOKS_API_KEY:
    st.warning("Google Books API key missing; covers and links may be limited.")
if OpenAI is None:
    st.info("Install the 'openai' package to enable the ChatGPT explanation feature.")
elif not OPENAI_API_KEY:
    st.info("Add OPENAI_API_KEY to .streamlit/secrets.toml to enable ChatGPT explanations.")


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
        title_col = (
            "track_name"
            if "track_name" in df.columns
            else "title_key"
            if "title_key" in df.columns
            else df.columns[0]
        )
        artist_col = (
            "track_artist"
            if "track_artist" in df.columns
            else "artist_key"
            if "artist_key" in df.columns
            else None
        )

        if artist_col:
            df["display_name"] = df[artist_col].astype(str) + " - " + df[title_col].astype(str)
        else:
            df["display_name"] = df[title_col].astype(str)

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_google_books_metadata(title: str, author: str = None) -> dict:
    """Look up a book on Google Books to enrich cards with cover and links."""
    if not title or not GOOGLE_BOOKS_API_KEY:
        return {}

    query = f"intitle:{title}"
    if author:
        query += f"+inauthor:{author}"

    params = {
        "q": query,
        "maxResults": 1,
        "printType": "books",
        "projection": "lite",
        "country": "FR",
        "key": GOOGLE_BOOKS_API_KEY,
    }

    try:
        resp = requests.get(
            "https://www.googleapis.com/books/v1/volumes",
            params=params,
            timeout=8,
        )
        resp.raise_for_status()
        items = resp.json().get("items") or []
        if not items:
            return {}

        volume_info = items[0].get("volumeInfo", {})
        sale_info = items[0].get("saleInfo", {})
        images = volume_info.get("imageLinks") or {}

        return {
            "title": volume_info.get("title"),
            "authors": volume_info.get("authors"),
            "thumbnail": images.get("thumbnail") or images.get("smallThumbnail"),
            "image": images.get("large") or images.get("medium") or images.get("small"),
            "infoLink": volume_info.get("infoLink"),
            "previewLink": volume_info.get("previewLink"),
            "canonicalVolumeLink": volume_info.get("canonicalVolumeLink"),
            "buy_link": sale_info.get("buyLink"),
        }
    except Exception:
        # Keep UI clean if Google Books is unreachable; fall back to placeholders.
        return {}

def _select_playlist_rows(songs_df: pd.DataFrame, playlist_ids: list[str]):
    """Subset helper that tolerates missing columns."""
    if songs_df is None or songs_df.empty or not playlist_ids:
        return pd.DataFrame(columns=songs_df.columns if songs_df is not None else [])

    if "songs_id" in songs_df.columns:
        subset = songs_df[songs_df["songs_id"].isin(playlist_ids)]
        if not subset.empty:
            return subset

    try:
        return songs_df.loc[playlist_ids]
    except Exception:
        try:
            return songs_df.iloc[playlist_ids]
        except Exception:
            return pd.DataFrame(columns=songs_df.columns)


ROLE_STATEMENT = (
    "Tu es un libraire/disquaire, qui doit expliquer le choix de nos recommendations "
    ", il faut que tu t'exprimes dans des termes simples, mais efficaces."
)

INSTRUCTION_LINES = [
    "Présente d'abord les chansons contenues dans la playlist, en donnant un très court descriptif de chaque chanson ou de la chanson, indique aussi le nombre de chansons dans la playlist.",
    "Explique pour chaque livre pourquoi il correspond à la playlist, en reliant explicitement des éléments musicaux et littéraires. Fais court.",
    "Présente ta réponse en français, un paragraphe (maximum une phrase) par livre classé par similarité décroissante.",
    "Termine avec une conclusion globale de pourquoi ces recommandations sont un bon choix pour cette playlist. Keep it simple, on parle d'emotion, de feelings, de topics qui serait correlé entre le choix des chansons, et el recommendation de livre",
]


def build_chatgpt_prompt(
    songs_df: pd.DataFrame, playlist_idx: list[str], recos: list[dict], model_label: str
) -> str:
    playlist_subset = _select_playlist_rows(songs_df, playlist_idx)
    playlist_subset = playlist_subset.fillna("Inconnu")

    song_lines = []
    for _, row in playlist_subset.iterrows():
        title = row.get("track_name") or row.get("title") or "Titre inconnu"
        artist = row.get("track_artist") or row.get("artist") or "Artiste inconnu"
        genre = row.get("genre")
        suffix = f" ({genre})" if isinstance(genre, str) and genre else ""
        song_lines.append(f"- {artist} – {title}{suffix}")

    recos_df = pd.DataFrame(recos or [])
    book_lines = []
    for rank, row in recos_df.reset_index(drop=True).iterrows():
        title = row.get("title") or row.get("book_title") or "Titre inconnu"
        author = row.get("author") or row.get("authors") or "Auteur inconnu"
        isbn = row.get("isbn") or row.get("ISBN")
        similarity = row.get("similarity") or row.get("similarity_score")
        desc = (row.get("description") or row.get("summary") or "").replace("\n", " ").strip()

        fragments = [f"{title} - {author}"]
        if isinstance(isbn, str) and isbn.strip():
            fragments.append(f"ISBN : {isbn.strip()}")
        if isinstance(similarity, (int, float)):
            fragments.append(f"Similarité : {similarity:.3f}")
        if desc:
            fragments.append(f"Résumé : {desc}")

        book_lines.append(f"{rank + 1}. " + " | ".join(fragments))

    song_section = "\n".join(song_lines) if song_lines else "- Playlist vide ou introuvable."
    book_section = "\n".join(book_lines) if book_lines else "- Aucune recommandation."
    instructions_section = "\n".join(f"- {line}" for line in INSTRUCTION_LINES)

    return (
        f"{ROLE_STATEMENT}\n\n"
        f"Playlist de départ ({len(playlist_subset)} chansons) :\n{song_section}\n\n"
        f"Livres recommandés par {model_label} :\n{book_section}\n\n"
        f"Consignes pour ChatGPT :\n{instructions_section}\n"
    )


def ask_chatgpt(prompt: str) -> str:
    if OpenAI is None or not OPENAI_API_KEY:
        return ""
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )
    return resp.output_text or ""

# --- Autosave/State Management ---
if "recommendations" not in st.session_state:
    st.session_state.recommendations = {"Albert": None, "Michel": None, "Tiffany": None}
if "last_songs" not in st.session_state:
    st.session_state.last_songs = []
if "explanations" not in st.session_state:
    st.session_state.explanations = {}

# --- Main App ---

# Sidebar
with st.sidebar:
    st.title("Model Selection")

    model_choice = st.radio(
        "Select Persona:",
        ("Albert", "Michel", "Tiffany"),
        index=0
    )

    st.divider()

    if model_choice == "Albert":
        st.image(
            "Media/thispersondoesnotexist4.jpeg",
            width=120,
        )
        st.markdown("### Albert 🤖")
        st.info(
            "I analyze the **context and meaning** of your songs using BERT embeddings "
            "to find books with a matching vibe."
        )
    elif model_choice == "Michel":
        st.image(
            "Media/thispersondoesnotexist3.jpeg",
            width=120,
        )
        st.markdown("### Michel 📊")
        st.info(
            "I look at the **numbers**—tempo, energy, valence. "
            "I'll find books that match the emotional curve of your playlist."
        )
    else:  # Tiffany
        st.image(
            "Media/thispersondoesnotexist2.jpeg",
            width=120,
        )
        st.markdown("### Tiffany 📝")
        st.info(
            "I find books by matching **keywords and lyrics** from your songs. "
            "I'm great at finding thematic connections!"
        )

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
        default=st.session_state.last_songs,  # restore selection if possible
    )

    if selected_display_names:
        # Optionnel : mémoriser les derniers morceaux choisis
        st.session_state.last_songs = selected_display_names

        selected_songs = df_songs[df_songs["display_name"].isin(selected_display_names)]
        input_ids = selected_songs["songs_id"].tolist()

        st.markdown("#### 2. Get Recommendations")

        if st.button(f"Ask {model_choice} to Recommend 🚀", type="primary"):
            with st.spinner(f"{model_choice} is thinking..."):
                try:
                    if model_choice == "Albert":
                        endpoint = f"{API_URL}/recommend/bert-big"
                        payload = {"song_ids": input_ids, "top_k": 3}  # Get a few more for grid
                    elif model_choice == "Michel":
                        endpoint = f"{API_URL}/recommend/numerical"
                        payload = {"song_ids": input_ids, "n_recommendations": 3}
                    else:  # Tiffany
                        endpoint = f"{API_URL}/recommend/tfidf"
                        payload = {"song_ids": input_ids, "n_recommendations": 3}

                    response = requests.post(endpoint, json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        enriched_results = []
                        for book in results:
                            title = book.get("title") or book.get("book_title")
                            author = book.get("author") or book.get("authors")
                            google_data = fetch_google_books_metadata(title, author)
                            if google_data:
                                book = {**book, "google_books": google_data}
                            enriched_results.append(book)
                        st.session_state.recommendations[model_choice] = enriched_results
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")

                except Exception as e:
                    st.error(f"Something went wrong: {e}")

    # --- Display Results ---
    current_results = st.session_state.recommendations.get(model_choice)

    if current_results:
        st.divider()
        st.markdown(f"### {model_choice}'s Picks for You")

        # Grid Layout
        cols = st.columns(3)
        for idx, book in enumerate(current_results):
            with cols[idx % 3]:
                title = book.get("title") or book.get("book_title")
                author = book.get("author") or "Unknown author"
                desc = book.get("description", "No description available.")
                score = book.get("similarity", 0) or book.get("similarity_score", 0)
                score_pct = f"{score:.0%}" if score <= 1 else f"{score:.2f}"
                google_info = book.get("google_books") or {}
                cover_url = (
                    google_info.get("thumbnail")
                    or google_info.get("image")
                    or book.get("image")
                    or "https://placehold.co/240x360?text=No+photo+on+google+books+api"
                )
                purchase_link = (
                    google_info.get("buy_link")
                    or google_info.get("infoLink")
                    or book.get("buy_link")
                )
                preview_link = google_info.get("preview_link") or google_info.get("previewLink")
                preview_length = 450
                if len(desc) > preview_length:
                    preview = desc[:preview_length].rsplit(" ", 1)[0] + "..."
                    desc_html = f"""
                        <div class='desc-text'>{preview}</div>
                        <details class='desc-toggle'>
                            <summary>Read more</summary>
                            <div class='desc-text'>{desc}</div>
                        </details>
                    """
                else:
                    desc_html = f"<div class='desc-text'>{desc}</div>"

                # Layout: Container -> Badge -> Title -> Author -> Spacer
                with st.container():
                    st.markdown(
                        f"""
                        <div class="book-card-container">
                            <div class="book-card-header">
                                <div class="cover-frame">
                                    <img src="{cover_url}" alt="Book cover for {title}" loading="lazy" />
                                </div>
                                <div class="card-meta">
                                    <div class="similarity-badge">{score_pct} Match</div>
                                    <div class="book-title" title="{title}">{title}</div>
                                    <div class="book-author">by {author}</div>
                                    <div class="google-note">Google Books cover & links appear here when available.</div>
                                    <div class="book-links">
                                        <a class="link-pill {'disabled' if not purchase_link else ''}" href="{purchase_link or 'javascript:void(0);'}" target="_blank" rel="noopener">Buy on Google Books</a>
                                        <a class="link-pill secondary {'disabled' if not preview_link else ''}" href="{preview_link or 'javascript:void(0);'}" target="_blank" rel="noopener">Read a sample</a>
                                    </div>
                                    {desc_html}
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.markdown("#### Explication par le libraire (ChatGPT)")
        if OpenAI is None:
            st.info("Installe la dépendance 'openai' pour activer l'explication automatique.")
        elif not OPENAI_API_KEY:
            st.info("Ajoute OPENAI_API_KEY dans les secrets pour demander une explication.")
        else:
            if st.button("Demander une explication sur ces recommandations", key=f"explain_{model_choice}"):
                prompt = build_chatgpt_prompt(
                    songs_df=df_songs,
                    playlist_idx=input_ids,
                    recos=current_results,
                    model_label=model_choice,
                )
                with st.spinner("Le libraire rédige ton explication..."):
                    try:
                        explanation = ask_chatgpt(prompt)
                        if explanation:
                            st.session_state.explanations[model_choice] = explanation
                        else:
                            st.warning("Aucune réponse reçue de ChatGPT.")
                    except Exception as e:
                        st.error(f"Erreur lors de l'appel à ChatGPT : {e}")

            if st.session_state.explanations.get(model_choice):
                st.markdown(st.session_state.explanations[model_choice])

    elif selected_display_names and not current_results:
        st.info(f"Click the button above to get recommendations from {model_choice}!")

else:
    st.info("Loading song database... if this takes too long, check the data path.")

# --- Developer Debugging ---
with st.sidebar:
    st.divider()
    with st.expander("🛠 Developer Debug"):
        st.write("Debug info for Google Books API:")
        
        # Check if key is present (don't show full key)
        if GOOGLE_BOOKS_API_KEY:
            masked_key = GOOGLE_BOOKS_API_KEY[:4] + "..." + GOOGLE_BOOKS_API_KEY[-4:]
            st.success(f"API Key loaded: {masked_key}")
        else:
            st.error("API Key NOT found in secrets!")

        # Test Button
        if st.button("Test Google Books API (The Alchemist)"):
            try:
                test_title = "The Alchemist"
                test_author = "Paulo Coelho"
                st.write(f"Fetching: {test_title} by {test_author}...")
                
                # Manual request to see raw output
                params = {
                    "q": f"intitle:{test_title}+inauthor:{test_author}",
                    "maxResults": 1,
                    "printType": "books",
                    "projection": "lite",
                    "country": "FR",
                    "key": GOOGLE_BOOKS_API_KEY,
                }
                resp = requests.get(
                    "https://www.googleapis.com/books/v1/volumes",
                    params=params,
                    timeout=10,
                )
                
                st.write(f"Status Code: `{resp.status_code}`")
                
                if resp.status_code == 200:
                    data = resp.json()
                    st.json(data)
                    items = data.get("items", [])
                    if items:
                        vol = items[0].get("volumeInfo", {})
                        st.image(vol.get("imageLinks", {}).get("thumbnail"), caption="Test Cover")
                    else:
                        st.warning("No items found.")
                else:
                    st.error(f"Error: {resp.text}")
                    
            except Exception as e:
                import traceback
                st.error(f"Exception during test: {e}")
                st.code(traceback.format_exc())
