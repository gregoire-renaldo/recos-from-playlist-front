# Fichier python permettant de faire une ponderation entre plusieurs routes d'api

from typing import List, Dict, Any, Optional
import pandas as pd
import requests

# On garde ces helpers tels quels si tu les as déjà
def _call_reco_api(
    url: str,
    playlist_ids: List[int],
    top_k: int,
    model_name: str,
    timeout: int = 30,
) -> pd.DataFrame:
    payload = {
        "playlist_ids": playlist_ids,
        "top_k": top_k,
    }

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        for key in ["results", "recommendations", "books", "items"]:
            if key in data and isinstance(data[key], list):
                df = pd.DataFrame(data[key])
                break
        else:
            df = pd.json_normalize(data)
    else:
        raise ValueError(f"Format de réponse inattendu pour {model_name}: {type(data)}")

    expected_cols = {"title", "author", "description", "isbn", "similarity"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans la réponse de {model_name}: {missing}. "
            f"Colonnes disponibles: {list(df.columns)}"
        )

    df = df[list(expected_cols)].copy()
    df["model"] = model_name
    return df


def _normalize_similarity_by_model(
    df: pd.DataFrame,
    col: str = "similarity",
    new_col: str = "similarity_norm",
) -> pd.DataFrame:
    df = df.copy()
    parts = []

    for model, group in df.groupby("model"):
        sims = group[col].astype(float)
        s_min, s_max = sims.min(), sims.max()
        if s_max > s_min:
            norm = (sims - s_min) / (s_max - s_min)
        else:
            norm = pd.Series(1.0, index=sims.index)

        group[new_col] = norm
        parts.append(group)

    return pd.concat(parts, ignore_index=True)


def get_ensemble_recommendations(
    playlist_ids: List[int],
    top_k_per_api: int,
    api_urls: Dict[str, str],
    top_k_final: int = 10,
    weights: Optional[Dict[str, float]] = None,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """
    Appelle N APIs de reco, combine leurs résultats avec une somme pondérée
    des similarités normalisées, et renvoie une liste de dict (JSON-like)
    contenant les recommandations finales.

    Args:
        playlist_ids: liste d'IDs de chansons (ou playlists).
        top_k_per_api: nombre de recommandations à demander à chaque API.
        api_urls: dict {model_name: api_url}
            ex: {"bert_en": ".../bert_en", "bert_ml": ".../bert_ml", ...}
        top_k_final: nombre de recommandations finales à renvoyer.
        weights: dict optionnel {model_name: poids}. Si None → poids égaux.
        timeout: timeout en secondes pour chaque appel d'API.

    Returns:
        List[dict]: chaque dict contient
            - isbn
            - title
            - author
            - description
            - score_final
            - models_contributing (liste des modèles où le livre apparaît)
    """
    if not api_urls:
        raise ValueError("api_urls ne doit pas être vide.")

    # 1️⃣ Appel de toutes les APIs fournies
    dfs = []
    for model_name, url in api_urls.items():
        df_m = _call_reco_api(
            url=url,
            playlist_ids=playlist_ids,
            top_k=top_k_per_api,
            model_name=model_name,
            timeout=timeout,
        )
        dfs.append(df_m)

    all_df = pd.concat(dfs, ignore_index=True)

    # 2️⃣ Normalisation des similarités par modèle
    df_norm = _normalize_similarity_by_model(all_df, col="similarity", new_col="similarity_norm")

    # 3️⃣ Poids : égaux par défaut si non fournis
    model_names = list(api_urls.keys())
    if weights is None:
        equal_weight = 1.0 / len(model_names)
        weights = {name: equal_weight for name in model_names}

    df_norm["weight"] = df_norm["model"].map(weights).fillna(0.0)

    # 4️⃣ Score pondéré par ligne
    df_norm["score_weighted"] = df_norm["similarity_norm"] * df_norm["weight"]

    # 5️⃣ Agrégation par livre (isbn)
    ensemble_df = (
        df_norm
        .groupby("isbn", as_index=False)
        .agg(
            score_final=("score_weighted", "sum"),
            title=("title", "first"),
            author=("author", "first"),
            description=("description", "first"),
            models_contributing=("model", lambda x: sorted(set(x))),
        )
        .sort_values("score_final", ascending=False)
    )

    # 6️⃣ On garde seulement le top_k_final
    top = ensemble_df.head(top_k_final)

    # 7️⃣ Conversion en JSON-like
    results: List[Dict[str, Any]] = []
    for _, row in top.iterrows():
        results.append(
            {
                "isbn": row["isbn"],
                "title": row["title"],
                "author": row["author"],
                "description": row["description"],
                "score_final": float(row["score_final"]),
                "models_contributing": list(row["models_contributing"]),
            }
        )

    return results
