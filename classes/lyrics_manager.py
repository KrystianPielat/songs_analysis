import logging
from .lyric_sources import LetrasMusSource, FandomSource, MakeItPersonalSource, GeniusLyricSource
from classes.log import setup_logging
import re

LOGGER = logging.getLogger(__name__)

class LyricsManager:
    """Class to handle lyrics fetching from various sources."""

    def __init__(self):
        self.providers = [
            LetrasMusSource(),
            MakeItPersonalSource(),
            FandomSource(),
            GeniusLyricSource()
        ]

    def fetch_lyrics(self, artist: str, title: str, clean_title: bool = True, clean_lyrics: bool = False):
        """Fetches the lyrics by trying providers in order."""
        if clean_title:
            title = LyricsManager._clean_title(title)
        for provider in self.providers:
            lyrics = provider.get_song_lyrics(artist, title)
            if lyrics:
                LOGGER.info(f"Lyrics found by {provider.__class__.__name__} for '{title}' by {artist}.")
                return LyricsManager._clean_lyrics(lyrics) if clean_lyrics else lyrics
            else:
                LOGGER.info(f"Lyrics not found by {provider.__class__.__name__} for '{title}' by {artist}.")
        LOGGER.warning(f"Lyrics not found for '{title}' by {artist} using all available providers.")
        return None


    @staticmethod
    def _clean_title(title: str):
        patterns = [
            r'\(.*?\)',                     # Remove parentheses
            r'\[.*?\]',                     # Remove square brackets
            r'(feat\.|ft\.|featuring)\s+[^,]+',  # Remove "feat." or "ft."
            r'\{.*?\}',                     # Remove curly braces
            r'\s*[-–—]\s*.*$',              # Remove text after dashes
            r'(acoustic|live|remix|edit|version|radio edit|original mix|karaoke|instrumental|clean|explicit|rework)\b.*$'  # Remove version indicators
        ]
        for pattern in patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE).strip()
        return title 

    @staticmethod
    def _clean_lyrics(text: str):
       # Use regex to remove all text inside square brackets, including the brackets
        clean_text = re.sub(r'\[.*?\]', '', text)
        
        clean_text = clean_text.strip()

        # Remove breaklines
        clean_text = clean_text.replace("\n", " ")
        return clean_text