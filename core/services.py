"""External API service clients for VibeWise."""

import logging
import requests
from youtubesearchpython import VideosSearch
from .config import ITUNES_TIMEOUT, YOUTUBE_LIMIT

# Set up logger
logger = logging.getLogger(__name__)


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
    msg = f"[YouTube Search] Query: '{query}'"
    logger.info(msg)
    print(msg)  # Backup print
    
    try:
        search = VideosSearch(query, limit=YOUTUBE_LIMIT)
        result = search.result()
        results = result.get("result", [])
        
        msg = f"[YouTube Search] Raw result keys: {list(result.keys())}"
        logger.info(msg)
        print(msg)
        
        msg = f"[YouTube Search] Found {len(results)} results"
        logger.info(msg)
        print(msg)
        
        if results:
            first = results[0]
            msg = f"[YouTube Search] First result keys: {list(first.keys())}"
            logger.info(msg)
            print(msg)
            
            thumbnail = first["thumbnails"][0]["url"]
            link = first["link"]
            
            msg = f"[YouTube Search] SUCCESS - Link: {link}"
            logger.info(msg)
            print(msg)
            return thumbnail, link
        else:
            msg = f"[YouTube Search] WARNING - No results for '{query}'"
            logger.warning(msg)
            print(msg)
            return None, None
    except Exception as e:
        msg = f"[YouTube Search] ERROR - {type(e).__name__}: {str(e)}"
        logger.error(msg)
        print(msg)
        import traceback
        traceback.print_exc()
        return None, None
