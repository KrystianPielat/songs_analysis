import os
import re
import urllib.request
import yt_dlp
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, error
import logging
from .lyrics_manager import LyricsManager
from classes.log import setup_logging
# setup_logging()


LOGGER = logging.getLogger(__name__)


yt_dlp_logger = logging.getLogger('yt_dlp')
yt_dlp_logger.setLevel(logging.WARNING)

class Song:
    """Class that represents a song and allows fetching its lyrics, downloading the song, and retrieving Spotify audio features."""
    
    def __init__(self, id, artist, title, album_art_url=None, popularity=None, explicit=None, album_release_year=None, duration_ms=None, save_path=None):
        self.id = id
        self.artist = self._format_request_param(artist)
        self.title = self._format_request_param(title)
        self.album_art_url = album_art_url
        self.lyrics = None
        self.popularity = popularity
        self.explicit = explicit
        self.album_release_year = album_release_year  # New attribute for album release year
        self.duration_ms = duration_ms  # New attribute for song duration
        self.save_path = save_path or os.getcwd()
        self.audio_features = None
        self.relevant_audio_features = {}
        self.lyrics_manager = LyricsManager()

    
    @staticmethod
    def _extract_album_release_year(album_data):
        """Extracts the year from the album's release date, handling different precisions."""
        release_date = album_data.get('release_date', None)
        release_precision = album_data.get('release_date_precision', 'year')
        
        if release_precision == 'year':
            return release_date  # The entire release date is just the year
        elif release_precision == 'month' or release_precision == 'day':
            return release_date.split('-')[0]  # Extract only the year part
        return None


    def fetch_spotify_features(self, spotify_client):
        """Fetches audio features from Spotify API and stores only relevant data."""
        if not self.id:
            LOGGER.warning(f"Song {self.title} by {self.artist} not found on Spotify.")
            return

        features = spotify_client.audio_features(self.id)
        if features:
            features = features[0]
            # Store only the relevant audio features
            self.relevant_audio_features = {
                'danceability': features.get('danceability'),
                'energy': features.get('energy'),
                'key': features.get('key'),
                'loudness': features.get('loudness'),
                'mode': features.get('mode'),
                'speechiness': features.get('speechiness'),
                'acousticness': features.get('acousticness'),
                'instrumentalness': features.get('instrumentalness'),
                'liveness': features.get('liveness'),
                'valence': features.get('valence'),
                'tempo': features.get('tempo'),
                'time_signature': features.get('time_signature')
            }
            LOGGER.info(f"Fetched Spotify audio features for {self.title} by {self.artist}: {self.relevant_audio_features}")
        else:
            LOGGER.warning(f"No audio features found for {self.title} by {self.artist}.")


    def fetch_lyrics(self):
        """Fetches the lyrics using the LyricsManager."""
        self.lyrics = self.lyrics_manager.fetch_lyrics(self.artist, self.title)
        return self.lyrics

    def download_song(self):
        """Searches YouTube for the song and downloads it as an MP3."""
        search_query = f"{self.artist} - {self.title}"
        LOGGER.info(f"Searching YouTube for: {search_query}")
        best_url = self._search_youtube(search_query)

        if not best_url:
            LOGGER.warning(f"No valid URL found for {search_query}. Skipping.")
            return

        if self.album_art_url:
            LOGGER.info(f"Downloading album art for {self.title}.")
            self._download_album_art()

        LOGGER.info(f"Downloading song: {search_query}")
        self._download_from_youtube(best_url)

        if self.album_art_url:
            LOGGER.info(f"Embedding album art into {self.title}.mp3")
            self._embed_album_art()

    def _search_youtube(self, query):
        """Searches YouTube and returns the best URL match."""
        try:
            from youtube_search import YoutubeSearch
            results_list = YoutubeSearch(query, max_results=1).to_dict()
            return f"https://www.youtube.com{results_list[0]['url_suffix']}"
        except IndexError:
            LOGGER.warning(f"No valid URLs found for {query}.")
        return None

    def _download_album_art(self):
        """Downloads the album art image from the given URL."""
        album_art_path = os.path.join(self.save_path, f"{self.title}.jpg")
        with open(album_art_path, 'wb') as f:
            f.write(urllib.request.urlopen(self.album_art_url).read())

    def _download_from_youtube(self, url):
        """Downloads the audio from a YouTube URL."""
        output_template = os.path.join(self.save_path, f'{self.title}.%(ext)s')
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'quiet': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }, {
                'key': 'FFmpegMetadata',
            }]
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    def _embed_album_art(self):
        """Embeds the downloaded album art into the MP3 file."""
        audio_file_path = os.path.join(self.save_path, f'{self.title}.mp3')
        album_art_path = os.path.join(self.save_path, f"{self.title}.jpg")
        audio = MP3(audio_file_path, ID3=ID3)
        try:
            audio.add_tags()
        except error:
            pass

        audio.tags.add(
            APIC(
                encoding=3,
                mime='image/jpeg',
                type=3,
                desc='Cover',
                data=open(album_art_path, 'rb').read()
            )
        )
        audio.save()
        os.remove(album_art_path)

    @staticmethod
    def _format_request_param(request_param):
        """Removes accents and special characters, and lowercases the input."""
        return request_param.lower()

    def __repr__(self):
        return f"<Song {self.title} by {self.artist}>"

    def __str__(self):
        return self.__repr__().replace("< ", "").replace(" >", "")
