import logging
from .lyric_sources import LetrasMusSource, FandomSource, MakeItPersonalSource, GeniusLyricSource
from classes.log import setup_logging
import re
from typing import Optional, List, Dict

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

    def fetch_lyrics(self, artist: str, title: str, clean_title: bool = True, clean_lyrics: bool = False) -> Optional[str]:
        """Fetches the lyrics by attempting to retrieve from providers in order.

        Args:
            artist (str): The artist's name.
            title (str): The song title.
            clean_title (bool, optional): Whether to clean the title to increase match accuracy. Defaults to True.
            clean_lyrics (bool, optional): Whether to clean the lyrics text for extra characters or annotations. Defaults to False.

        Returns:
            Optional[str]: The lyrics if found, or None if no lyrics are available from any provider.
        """
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
    def _clean_title(title: str) -> str:
        """Cleans the song title by removing unnecessary parts such as featured artists and version information.

        Args:
            title (str): The original song title.

        Returns:
            str: The cleaned song title.
        """
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
    def _clean_lyrics(text: str) -> str:
        """Cleans the lyrics text by removing unwanted annotations and line breaks.

        Args:
            text (str): The original lyrics text.

        Returns:
            str: The cleaned lyrics text.
        """
       # Use regex to remove all text inside square brackets, including the brackets
        clean_text = re.sub(r'\[.*?\]', '', text)
        
        clean_text = clean_text.strip()

        # Remove breaklines
        clean_text = clean_text.replace("\n", " ")
        return clean_text