"""External API service clients for VibeWise."""

import requests
from youtubesearchpython import VideosSearch
from .config import ITUNES_TIMEOUT, YOUTUBE_LIMIT


def get_itunes_cover(song, artist=None):
    """Fetch album cover art from iTunes API.
    
    Args:
        song: Song name
        artist: Artist name (optional)
        
    Returns:
        URL to cover art image (600x600) or None if not found
    """
    query = song + (f" {artist}" if artist else "")
    try:
        response = requests.get(
            f"https://itunes.apple.com/search?term={query}&limit=1",
            timeout=ITUNES_TIMEOUT
        ).json()
        if response["resultCount"]:
            return response["results"][0]["artworkUrl100"].replace("100x100", "600x600")
    except Exception:
        return None
    return None


def get_youtube_video(song, artist=None):
    """Search for YouTube video of a song.
    
    Args:
        song: Song name
        artist: Artist name (optional)
        
    Returns:
        Tuple of (thumbnail_url, video_link) or (None, None) if not found
    """
    query = song + (f" {artist}" if artist else "")
    try:
        results = VideosSearch(query, limit=YOUTUBE_LIMIT).result()["result"]
        if results:
            return results[0]["thumbnails"][0]["url"], results[0]["link"]
    except Exception:
        return None, None
    return None, None
