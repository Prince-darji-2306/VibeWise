import time
import logging
import numpy as np
from PIL import Image
import streamlit as st
from deepface import DeepFace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from core.models import (load_embedding_model,
                         load_faiss_index,
                         load_song_metadata)

from core.recommender import (get_recommendations,
                             enrich_recommendations_parallel)
                             
from core.config import (EMOTION_MAPPING,
                         HAPPY_CONFIDENCE_THRESHOLD,
                         SAD_CONFIDENCE_THRESHOLD)  

# Initializing session state
if "mode" not in st.session_state:
    st.session_state.mode = "Set Vibe"
if "results" not in st.session_state:
    st.session_state.results = []
if "video_url" not in st.session_state:
    st.session_state.video_url = ""
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "user_mood" not in st.session_state:
    st.session_state.user_mood = None

st.set_page_config(page_title="VibeWise | Discover Your Next Favorite Song", layout="wide",page_icon='static/img/icon.png')


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("static/css/mstyle.css")

st.sidebar.markdown("## Navigation")
logger.info(f"[Navigation] Current mode: {st.session_state.mode}, video_url: {st.session_state.video_url}")
if st.sidebar.button("Set Vibe 🎧"):
    st.session_state.mode = "Set Vibe"
if st.sidebar.button("Song 🎬"):
    logger.info(f"[Navigation] Song button clicked, video_url: '{st.session_state.video_url}'")
    if st.session_state.video_url:
        st.session_state.mode = "Song"
    else:
        st.warning("No video selected!")
if st.sidebar.button("Detect Mood 🎭"):
    st.session_state.mode = "Detect Mood"

# ============================
# MODE: SET VIBE
# ============================
if st.session_state.mode == "Set Vibe":
    st.markdown("<h1 style='padding-bottom:0px;'>🎶 Song Recommendation</h1>", unsafe_allow_html=True)
    col2, col3 = st.columns([6,2])

    with col2:
        query_input = st.text_input(
            label='Set Your VIBE😉',
            key="query_input",
            placeholder="Song or Artist Name....",
            label_visibility="hidden"
        )
        
        df = load_song_metadata()
        matches = []
        if len(query_input) > 3:
            matches = df[df['searchq'].str.lower().str.startswith(query_input.lower())]['searchq'].unique().tolist()

        if matches:
            selection = st.selectbox("Did you mean:", ['No'] + matches, key="suggestions")
            if selection != 'No':
                query_input = selection

    with col3:

        st.markdown("<div class='mbutton'></div>", unsafe_allow_html=True)
        if st.button("Recommend", use_container_width=True) and query_input.strip() != '' or st.session_state.user_mood:
            with st.spinner("Setting Vibe..."):
                model = load_embedding_model()
                index = load_faiss_index()

                recs = get_recommendations(query_input, model, index, df)
                results = enrich_recommendations_parallel(recs)

                st.session_state.user_mood = None
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
            logger.info(f"[UI] Song {i}: {r['song']} by {r['artist']}, link: {r.get('link', 'None')}")
            if st.button(f"▶ Watch Now", key=f"song_{i}"):
                logger.info(f"[UI] Watch Now clicked for song {i}, setting video_url: '{r.get('link')}'")
                st.session_state.video_url = r["link"]
                logger.info(f"[UI] Mode changing to Song, video_url now: '{st.session_state.video_url}'")
                st.session_state.mode = "Song"
                st.rerun()

# ============================
# MODE: SONG VIEW
# ============================
elif st.session_state.mode == "Song":
    logger.info(f"[Song Mode] video_url: '{st.session_state.video_url}'")
    st.markdown("<h1>🎬 Now Playing</h1>", unsafe_allow_html=True)
    if st.session_state.video_url:
        logger.info(f"[Song Mode] Attempting to play video: {st.session_state.video_url}")
        st.markdown('<div class="custom-video-container">', unsafe_allow_html=True)
        st.video(st.session_state.video_url)
        st.markdown('</div>', unsafe_allow_html=True)
        logger.info("[Song Mode] Video player rendered successfully")
    else:
        logger.error("[Song Mode] No video URL available to play!")
        st.warning("No video selected.")
    if st.button("🔙 Back to Set Vibe"):
        st.session_state.mode = "Set Vibe"
        st.rerun()

# ============================
# MODE: DETECT MOOD
# ============================
elif st.session_state.mode == "Detect Mood":
    st.markdown("<h1>🎭Mood Detection</h1>", unsafe_allow_html=True)
    st.write("""
    - **Click “Photo”** to check your mood.  
    """)

    img_file_buffer = st.camera_input("Take a photo")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)

        with st.spinner("Analyzing your mood..."):
            try:
                result = DeepFace.analyze(
                    img_array,
                    actions=['emotion'],
                    enforce_detection=True
                )[0]

                detected_emotion = result['dominant_emotion']
                confidence = result['emotion'][detected_emotion]

                song_emotion = EMOTION_MAPPING.get(detected_emotion.lower(), 'chill')

                if song_emotion == 'happy' and confidence < HAPPY_CONFIDENCE_THRESHOLD:
                    song_emotion = 'romantic'
                elif song_emotion == 'sad' and confidence < SAD_CONFIDENCE_THRESHOLD:
                    song_emotion = 'chill'

                if song_emotion:
                    st.subheader("🎶 Detected Vibe:")
                    st.success(f" Feeling **{detected_emotion}** ? Let's Play **{song_emotion}** Songs..😉")
                    st.session_state.query_input = song_emotion + ' songs'
                    st.session_state.user_mood = True
                    st.session_state.mode = 'Set Vibe'
                    time.sleep(3)
                    st.rerun()

            except Exception as e:
                st.error(f"Face analysis failed: {str(e)}")

