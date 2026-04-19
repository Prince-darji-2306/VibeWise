"""Model loading and caching utilities for VibeWise."""

import os
import faiss
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download, snapshot_download
from sentence_transformers import SentenceTransformer

from .config import MODEL_REPO, INDEX_FILENAME, CSV_FILENAME, MODEL_PATH


@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence transformer model from Hugging Face."""
    snapshot_dir = snapshot_download(
        repo_id=MODEL_REPO,
        repo_type="model",
        local_dir="hf_model",
        allow_patterns=[f"{MODEL_PATH}/*"]
    )
    model_dir = os.path.join(snapshot_dir, MODEL_PATH)
    # model_dir = 'model/song_recommender_model'
    return SentenceTransformer(model_dir)


@st.cache_resource
def load_faiss_index():
    """Load and cache the FAISS index for similarity search."""
    index_path = hf_hub_download(repo_id=MODEL_REPO, filename=INDEX_FILENAME)
    # index_path = 'model/song_index.faiss'
    return faiss.read_index(index_path)


@st.cache_data
def load_song_metadata():
    """Load and cache the song metadata DataFrame."""
    csv_path = hf_hub_download(repo_id=MODEL_REPO, filename=CSV_FILENAME)
    # csv_path = 'model/song_metadata.csv'  
    df = pd.read_csv(csv_path)
    if 'searchq' not in df.columns:
        df['searchq'] = (df['song'] + " by " + df['artist']).str.strip()
    return df
