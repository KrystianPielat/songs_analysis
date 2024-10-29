import spotipy
import spotipy.oauth2 as oauth2
import time
import logging
from requests.exceptions import ConnectionError, HTTPError, ReadTimeout

LOGGER = logging.getLogger(__name__)

class SpotifyManager:
    """Class to handle Spotify API interactions with retry logic."""

    def __init__(self, client_id, client_secret, max_retries=5, base_delay=1):
        auth_manager = oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        self.client = spotipy.Spotify(auth_manager=auth_manager)
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _make_request(self, func, *args, **kwargs):
        """Helper method to make requests with retry logic and exponential backoff."""
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except (ConnectionError, HTTPError, ReadTimeout) as e:
                LOGGER.warning(f"Attempt {attempt} failed with error: {e}")
                
                if attempt < self.max_retries:
                    # Non-linear exponential backoff delay between retries
                    delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                    LOGGER.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    LOGGER.error(f"Failed after {self.max_retries} attempts.")
                    raise

    def get_playlist(self, uri):
        """Fetches playlist metadata from Spotify."""
        return self._make_request(self.client.playlist, uri)

    def get_audio_features(self, song_id):
        """Fetches audio features from Spotify."""
        return self._make_request(self.client.audio_features, song_id)

    def get_playlist_tracks(self, uri):
        """Fetches tracks from a Spotify playlist with pagination."""
        tracks = []
        playlist = self.get_playlist(uri)
        results = playlist['tracks']
        while results:
            tracks.extend(results['items'])
            results = self._make_request(self.client.next, results) if results['next'] else None
        return [track['track'] for track in tracks if track.get('track') is not None]

    def get_artist(self, artist_id):
        """Fetches artist data from Spotify."""
        return self._make_request(self.client.artist, artist_id)
