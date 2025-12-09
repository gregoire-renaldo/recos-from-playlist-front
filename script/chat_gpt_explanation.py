"""Utilities to build the ChatGPT comparative prompt for playlist/book recos."""
from __future__ import annotations

from typing import Iterable, List, Tuple, Any
import pandas as pd


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, pd.Series)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return [value]


def _select_playlist_rows(songs_df: pd.DataFrame, playlist_idx: Any) -> pd.DataFrame:
    """Return the subset of songs_df corresponding to the playlist selection."""
    if songs_df is None or songs_df.empty:
        return pd.DataFrame()

    ids = _ensure_list(playlist_idx)
    if not ids:
        return pd.DataFrame(columns=songs_df.columns)

    # If ids look like song_ids (strings), try matching on songs_id column.
    if "songs_id" in songs_df.columns and any(isinstance(x, str) for x in ids):
        subset = songs_df[songs_df["songs_id"].isin(ids)]
        if not subset.empty:
            return subset

    # Fallback: treat as positional indices.
    try:
        return songs_df.iloc[ids]
    except Exception:
        try:
            return songs_df.loc[ids]
        except Exception:
            return pd.DataFrame(columns=songs_df.columns)


def _format_playlist(playlist_df: pd.DataFrame) -> str:
    if playlist_df is None or playlist_df.empty:
        return "- (playlist vide)"
    lines = []
    for _, row in playlist_df.iterrows():
        artist = row.get("track_artist") or "Artiste inconnu"
        title = row.get("track_name") or row.get("title") or "Titre inconnu"
        genre = row.get("genre") or "genre ?"
        lines.append(f"- {artist} – {title} ({genre})")
    return "\n".join(lines)


def _format_reco_item(book: dict) -> str:
    title = book.get("title") or book.get("book_title") or "Titre inconnu"
    author = book.get("author") or book.get("authors") or "Auteur inconnu"
    score = book.get("similarity") or book.get("similarity_score")
    score_txt = f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"
    desc = book.get("description") or book.get("summary") or ""
    desc = str(desc).strip() if desc else ""
    if desc and len(desc) > 180:
        desc = desc[:180].rsplit(" ", 1)[0] + "..."
    return f"- {title} — {author} (score: {score_txt}) : {desc}"


def _format_recos_by_model(
    recos_by_model: Iterable[Tuple[str, Any]], max_per_model: int = 5
) -> str:
    parts = []
    for label, recos in recos_by_model:
        if recos is None:
            continue
        if isinstance(recos, pd.DataFrame):
            items = recos.head(max_per_model).to_dict("records")
        else:
            items = list(recos)[:max_per_model]
        if not items:
            continue
        lines = "\n".join(_format_reco_item(b) for b in items)
        parts.append(f"{label} :\n{lines}")
    return "\n\n".join(parts) if parts else "- aucune recommandation disponible."


def build_comparative_prompt(
    songs_df: pd.DataFrame,
    playlist_idx: Any,
    recos_by_model: Iterable[Tuple[str, Any]],
    max_per_model: int = 5,
) -> str:
    """Build the French prompt sent to ChatGPT for the comparative analysis."""
    playlist_df = _select_playlist_rows(songs_df, playlist_idx)
    playlist_block = _format_playlist(playlist_df)
    recos_block = _format_recos_by_model(recos_by_model, max_per_model)

    return f"""Tu es un libraire expert qui compare des recommandations generees par plusieurs modeles.
Ta mission: commenter et synthetiser en francais les recos les plus pertinentes par rapport a la playlist.

Playlist:
{playlist_block}

Recommandations par modele:
{recos_block}

Consignes:
- Compare les modeles: releve les recos fortes et les divergences.
- Cite 2-3 recos pertinentes par modele avec une raison courte.
- Signale les suggestions douteuses ou hors-sujet.
- Termine par 3 livres a prioriser (tous modeles confondus) avec justification concise.
Repond en moins de 180 mots, ton: conseil de libraire, clair et direct."""
