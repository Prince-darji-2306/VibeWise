import os
import faiss
import requests
import streamlit as st
import pandas as pd
from sklearn.preprocessing import normalize
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from youtubesearchpython import VideosSearch
from concurrent.futures import ThreadPoolExecutor, as_completed

CONFIG = st.secrets['MODEL']

path = snapshot_download(
    repo_id=CONFIG,
    repo_type="model",
    local_dir="hf_model",
    allow_patterns=["model/song_recommender_model/*"]
)

MODEL_PATH = os.path.join(path, "model", "song_recommender_model")
INDEX_PATH = hf_hub_download(repo_id=CONFIG, filename="model/song_index.faiss")
CSV_PATH = hf_hub_download(repo_id=CONFIG, filename="model/song_metadata.csv")

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_PATH)

@st.cache_resource
def load_index():
    return faiss.read_index(INDEX_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    if 'searchq' not in df.columns:
        df['searchq'] = (df['song'] + " " + df['artist']).str.strip().str.lower()
    return df

def recommend(query, top_k=5):
    emb = normalize(model.encode([query]))
    _, I = index.search(emb, top_k)
    return df.iloc[I[0]]

def get_cover(song, artist=None):
    q = song + (f" {artist}" if artist else "")
    try:
        r = requests.get(f"https://itunes.apple.com/search?term={q}&limit=1", timeout=2).json()
        if r["resultCount"]:
            return r["results"][0]["artworkUrl100"].replace("100x100", "600x600")
    except:
        return None
    return None

def get_youtube(song, artist=None):
    q = song + (f" {artist}" if artist else "")
    try:
        results = VideosSearch(q, limit=1).result()["result"]
        if results:
            return results[0]["thumbnails"][0]["url"], results[0]["link"]
    except:
        return None, None
    return None, None

# Initializing session state
if "mode" not in st.session_state:
    st.session_state.mode = "Set Vibe"
if "results" not in st.session_state:
    st.session_state.results = []
if "video_url" not in st.session_state:
    st.session_state.video_url = ""

st.set_page_config(page_title="VibeWise | Discover Your Next Favorite Song", layout="wide",page_icon='static/img/icon.png')


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("static/css/mstyle.css")

st.sidebar.markdown("## Navigation")
if st.sidebar.button("Set Vibe 🎧"):
    st.session_state.mode = "Set Vibe"
if st.sidebar.button("Song 🎬"):
    if st.session_state.video_url:
        st.session_state.mode = "Song"
    else:
        st.warning("No video selected!")

# ============================
# MODE: SET VIBE
# ============================
if st.session_state.mode == "Set Vibe":
    st.markdown("<h1 style='padding-bottom:0px;'>🎶 Song Recommendation</h1>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([1,6,2,1])

    with col2:
        query_input = st.text_input(
            label='Set Your VIBE😉',
            key="query_input",
            placeholder="Song or Artist Name....",
            label_visibility="hidden"
        )
        
        df = load_data()
        matches = []
        if len(query_input) > 3:
            matches = df[df['searchq'].str.lower().str.startswith(query_input.lower())]['searchq'].unique().tolist()

        if matches:
            selection = st.selectbox("Did you mean:", ['No'] + matches, key="suggestions")
            if selection != 'No':
                query_input = selection

    with col3:

        st.markdown("<div class='mbutton'></div>", unsafe_allow_html=True)
        if st.button("Recommend", use_container_width=True) and query_input.strip() != '':
            with st.spinner("Setting Vibe..."):
                model = load_model()
                index = load_index()

                recs = recommend(query_input)

                def enrich(r):
                    cover = get_cover(r['song'], r['artist']) or None
                    thumb, yt_link = get_youtube(r['song'], r['artist'])
                    return {
                        "song": r["song"],
                        "artist": r["artist"],
                        "text": r["text"],
                        "cover": cover or thumb,
                        "link": yt_link
                    }

                with ThreadPoolExecutor(max_workers=5) as ex:
                    futures = [ex.submit(enrich, row) for _, row in recs.iterrows()]
                    results = [f.result() for f in as_completed(futures)]

                st.session_state.results = results

    # Show results
    cols = st.columns(3)
    for i, r in enumerate(st.session_state.results):
        with cols[i % 3]:
            st.markdown(f"""
                <div class="card">
                    <img src="{r['cover']}" class="song-image" />
                    <div class="card-text">
                        <div class="song-title">{r['song']}</div>
                        <div class="artist-name">By {r['artist']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            if st.button(f"▶ Watch Now", key=f"song_{i}"):
                st.session_state.video_url = r["link"]
                st.session_state.mode = "Song"
                st.rerun()

# ============================
# MODE: SONG VIEW
# ============================
elif st.session_state.mode == "Song":
    st.markdown("<h1>🎬 Now Playing</h1>", unsafe_allow_html=True)
    if st.session_state.video_url:
        st.video(st.session_state.video_url)
    else:
        st.warning("No video selected.")
    if st.button("🔙 Back to Set Vibe"):
        st.session_state.mode = "Set Vibe"
        st.rerun()
