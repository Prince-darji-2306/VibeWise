"""Core recommendation engine for VibeWise."""

import logging
from sklearn.preprocessing import normalize
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import DEFAULT_TOP_K, MAX_WORKERS, EMOTION_MAPPING, HAPPY_CONFIDENCE_THRESHOLD, SAD_CONFIDENCE_THRESHOLD
from .services import get_itunes_cover, get_youtube_video

logger = logging.getLogger(__name__)


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
    song = row['song']
    artist = row['artist']
    logger.info(f"[Enrich] Processing: '{song}' by '{artist}'")
    
    cover = get_itunes_cover(song, artist)
    logger.info(f"[Enrich] iTunes cover result: {cover is not None}")
    
    thumbnail, yt_link = get_youtube_video(song, artist)
    logger.info(f"[Enrich] YouTube thumbnail: {thumbnail is not None}, link: {yt_link is not None}")
    
    result = {
        "song": song,
        "artist": artist,
        "text": row["text"],
        "cover": cover or thumbnail,
        "link": yt_link
    }
    logger.info(f"[Enrich] Final result link: {result['link']}")
    return result


def enrich_recommendations_parallel(recommendations_df):
    """Enrich multiple song recommendations with metadata in parallel.
    
    Args:
        recommendations_df: DataFrame of recommended songs
        
    Returns:
        List of enriched song dictionaries
    """
    logger.info(f"[Parallel Enrich] Starting enrichment for {len(recommendations_df)} songs")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(enrich_song_data, row)
            for _, row in recommendations_df.iterrows()
        ]
        results = [future.result() for future in as_completed(futures)]
    
    # Log summary of links
    links_found = sum(1 for r in results if r.get('link'))
    logger.info(f"[Parallel Enrich] Completed: {links_found}/{len(results)} songs have video links")
    for i, r in enumerate(results):
        logger.info(f"[Parallel Enrich] Result {i}: {r['song']} - link: {r.get('link', 'None')}")
    
    return results


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
