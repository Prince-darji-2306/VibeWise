"""Core recommendation engine for VibeWise."""

from sklearn.preprocessing import normalize
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import DEFAULT_TOP_K, MAX_WORKERS, EMOTION_MAPPING, HAPPY_CONFIDENCE_THRESHOLD, SAD_CONFIDENCE_THRESHOLD
from .services import get_itunes_cover, get_youtube_video


def get_recommendations(query, model, index, df, top_k=DEFAULT_TOP_K):
    """Get song recommendations based on query embedding similarity.
    
    Args:
        query: Search query string
        model: SentenceTransformer model for encoding
        index: FAISS index for similarity search
        df: Song metadata DataFrame
        top_k: Number of recommendations to return
        
    Returns:
        DataFrame with top_k recommended songs
    """
    embedding = normalize(model.encode([query]))
    _, indices = index.search(embedding, top_k)
    return df.iloc[indices[0]]


def enrich_song_data(row):
    """Enrich a song record with cover art and YouTube link.
    
    Args:
        row: DataFrame row with song and artist information
        
    Returns:
        Dictionary with enriched song data
    """
    cover = get_itunes_cover(row['song'], row['artist'])
    thumbnail, yt_link = get_youtube_video(row['song'], row['artist'])
    return {
        "song": row["song"],
        "artist": row["artist"],
        "text": row["text"],
        "cover": cover or thumbnail,
        "link": yt_link
    }


def enrich_recommendations_parallel(recommendations_df):
    """Enrich multiple song recommendations with metadata in parallel.
    
    Args:
        recommendations_df: DataFrame of recommended songs
        
    Returns:
        List of enriched song dictionaries
    """
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(enrich_song_data, row)
            for _, row in recommendations_df.iterrows()
        ]
        return [future.result() for future in as_completed(futures)]


def get_mood_based_query(detected_emotion, confidence):
    """Convert detected emotion to a song search query.
    
    Args:
        detected_emotion: Emotion string from DeepFace
        confidence: Confidence score (0-100)
        
    Returns:
        Search query string for mood-based recommendations
    """
    song_emotion = EMOTION_MAPPING.get(detected_emotion.lower(), 'chill')
    
    # Apply confidence-based adjustments
    if song_emotion == 'happy' and confidence < HAPPY_CONFIDENCE_THRESHOLD:
        song_emotion = 'romantic'
    elif song_emotion == 'sad' and confidence < SAD_CONFIDENCE_THRESHOLD:
        song_emotion = 'chill'
    
    return song_emotion + ' songs'
