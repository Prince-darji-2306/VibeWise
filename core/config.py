"""Configuration settings for VibeWise application."""

import streamlit as st

# Model configuration from Streamlit secrets
MODEL_REPO = st.secrets.get('MODEL', 'Prince-2025/VibeWise-Model')

# File paths within the Hugging Face repository
INDEX_FILENAME = "model/song_index.faiss"
CSV_FILENAME = "model/song_metadata.csv"
MODEL_PATH = "model/song_recommender_model"

# Recommendation settings
DEFAULT_TOP_K = 5
MAX_WORKERS = 5

# External API settings
ITUNES_TIMEOUT = 2
YOUTUBE_LIMIT = 1

# Emotion mapping configuration
EMOTION_MAPPING = {
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'neutral': 'chill',
    'surprise': 'energetic',
    'fear': 'motivational',
    'disgust': 'romantic'
}

# Confidence thresholds
HAPPY_CONFIDENCE_THRESHOLD = 85
SAD_CONFIDENCE_THRESHOLD = 80
