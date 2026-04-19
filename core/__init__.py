"""Core modules for VibeWise application."""

from .config import (
    MODEL_REPO,
    INDEX_FILENAME,
    CSV_FILENAME,
    MODEL_PATH,
    DEFAULT_TOP_K,
    MAX_WORKERS,
    ITUNES_TIMEOUT,
    YOUTUBE_LIMIT,
    EMOTION_MAPPING,
    HAPPY_CONFIDENCE_THRESHOLD,
    SAD_CONFIDENCE_THRESHOLD,
)
from .models import load_embedding_model, load_faiss_index, load_song_metadata
from .services import get_itunes_cover, get_youtube_video
from .recommender import (
    get_recommendations,
    enrich_song_data,
    enrich_recommendations_parallel,
    get_mood_based_query,
)

__all__ = [
    "MODEL_REPO",
    "INDEX_FILENAME",
    "CSV_FILENAME",
    "MODEL_PATH",
    "DEFAULT_TOP_K",
    "MAX_WORKERS",
    "ITUNES_TIMEOUT",
    "YOUTUBE_LIMIT",
    "EMOTION_MAPPING",
    "HAPPY_CONFIDENCE_THRESHOLD",
    "SAD_CONFIDENCE_THRESHOLD",
    "load_embedding_model",
    "load_faiss_index",
    "load_song_metadata",
    "get_itunes_cover",
    "get_youtube_video",
    "get_recommendations",
    "enrich_song_data",
    "enrich_recommendations_parallel",
    "get_mood_based_query",
]
