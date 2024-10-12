import requests
import unicodedata
import re
import os
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
import logging
LOGGER = logging.getLogger(__name__)


class LyricSource(ABC):
    """Abstract base class for a lyric source provider."""

    @abstractmethod
    def get_song_lyrics(self, artist, title):
        """Fetches the lyrics for the given song and artist."""
        pass

class MusixmatchSource(LyricSource):
    """Musixmatch lyrics provider implementation."""

    def get_song_lyrics(self, artist, title):
        """Fetches the lyrics for the given song and artist from Musixmatch."""
        search_url = f"https://www.musixmatch.com/search/{artist} {title}"
        try:
            search_page = requests.get(search_url)
            search_soup = BeautifulSoup(search_page.text, 'html.parser')
            
            # Find the first song result
            song_link = search_soup.find("a", {"class": "title"})
            if song_link:
                song_url = "https://www.musixmatch.com" + song_link['href']
                return self._scrape_lyrics(song_url)
            else:
                return "Lyrics not found."
        except requests.RequestException:
            return "Lyrics not found."  # Handle network or search issues

    def _scrape_lyrics(self, url):
        """Scrapes lyrics from a specific Musixmatch song page."""
        try:
            song_page = requests.get(url)
            soup = BeautifulSoup(song_page.text, 'html.parser')
            
            # Look for the div that contains the lyrics
            lyrics_div = soup.find("div", {"class": "mxm-lyrics__content"})
            if lyrics_div:
                return lyrics_div.get_text(separator="\n").strip()
            else:
                return "Lyrics not found."
        except requests.RequestException:
            return "Lyrics not found."  # Handle network or scraping issues


class GeniusLyricSource(LyricSource):
    def __init__(self):
        self.base_url = "https://api.genius.com"
        self.headers = {'Authorization': f'Bearer {os.environ.get("GENIUS_API_TOKEN")}'}
    
    def get_song_lyrics(self, artist, title):
        """Fetches the lyrics for the given song and artist using the Genius API."""
        song_info = self._search_song(title, artist)
        if song_info:
            song_url = song_info['url']
            LOGGER.debug("Song url: " + str(song_url))
            lyrics = self._scrape_lyrics(song_url)
            return lyrics
        return None
    
    def _search_song(self, song_title, artist_name=None):
        search_url = f"{self.base_url}/search"
        data = {'q': song_title}
        response = requests.get(search_url, headers=self.headers, params=data)
        LOGGER.debug("_search_song response: " + str(response.status_code))
        if response.status_code == 200:
            json_data = response.json()
            LOGGER.debug("hits: " + str(len(json_data['response']['hits'])))
            for hit in json_data['response']['hits']:
                if artist_name and artist_name.lower() not in hit['result']['primary_artist']['name'].lower():
                    LOGGER.debug(f"Searched vs found artist name mismatch: searched: {artist_name.lower()}  |  found: {hit['result']['primary_artist']['name'].lower()}")
                    continue
                return hit['result']
        else:
            LOGGER.error("Failed to fetch lyrics from Genius API: " + str(response.content))
        return None


    def _scrape_lyrics(self, url):
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        lyrics_divs = soup.select("[data-lyrics-container=true]")
        LOGGER.debug("_scrape_lyrics divs: " + str(lyrics_divs))
        texts = []
        for div in lyrics_divs:
            texts.append(div.get_text(separator='\n').strip())
        return "\n".join(texts) if texts else None



class LetrasMusSource(LyricSource):
    """Letras.mus.br lyrics provider implementation."""

    def get_song_lyrics(self, artist, title):
        url = f"https://www.letras.mus.br/{artist}/{title}"
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.RequestException:
            return ""  # Return empty if there's a network issue or 404

        soup = BeautifulSoup(response.text, "html.parser")
        lyric_box = soup.find("div", {"class": "cnt-letra p402_premium"})

        if not lyric_box:
            return ""

        replacements = {
            '<div class="cnt-letra p402_premium">': "",
            "</div>": "",
            "<br/>": "\n",
            "<br>": "\n",
            "</br>": "\n",
            "</p><p>": "\n\n",
            "</p>": "",
            "<p>": ""
        }

        lyrics = str(lyric_box)
        for old, new in replacements.items():
            lyrics = lyrics.replace(old, new)

        return lyrics.strip()


class MakeItPersonalSource(LyricSource):
    """MakeItPersonal.co lyrics provider implementation."""

    def get_song_lyrics(self, artist, title):
        pageurl = f"https://makeitpersonal.co/lyrics?artist={artist}&title={title}"
        try:
            lyrics = requests.get(pageurl).text.strip()
        except requests.RequestException:
            return ""  # Return empty if there's a network issue

        if lyrics.startswith("Sorry"):
            return ""

        return lyrics


class FandomSource(LyricSource):
    """Fandom.com lyrics provider implementation."""

    def get_song_lyrics(self, artist, title):
        wiki_url = "https://lyrics.fandom.com/wiki/"
        title = title.replace(" ", "_")
        artist = artist.replace(" ", "_")
        url = wiki_url + f"{artist}:{title}"

        try:
            r = requests.get(url)
            r.raise_for_status()
        except requests.RequestException:
            return ""  # Return empty if there's a network issue

        soup = BeautifulSoup(r.text, "html.parser")
        lyric_box = soup.find("div", {"class": "lyricbox"})

        if not lyric_box:
            return ""

        replacements = {
            "<br/>": "\n",
            '<div class="lyricbox">': "",
            '<div class="lyricsbreak">': "",
            "</div>": ""
        }

        lyrics = str(lyric_box)
        for old, new in replacements.items():
            lyrics = lyrics.replace(old, new)

        return lyrics.strip()

