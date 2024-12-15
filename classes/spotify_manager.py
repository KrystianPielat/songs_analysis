import spotipy
import spotipy.oauth2 as oauth2
import time
import logging
from requests.exceptions import ConnectionError, HTTPError, ReadTimeout
from typing import Any, Dict, Optional, List

LOGGER = logging.getLogger(__name__)

class SpotifyManager:
    """Class to handle Spotify API interactions with retry logic and enhanced functionality."""

    def __init__(self, client_id: str, client_secret: str, max_retries: int = 5, base_delay: int = 1) -> None:
        """
        Initializes the SpotifyManager with API credentials and retry configuration.

        Args:
            client_id (str): Spotify API client ID.
            client_secret (str): Spotify API client secret.
            max_retries (int, optional): Maximum number of retry attempts. Defaults to 5.
            base_delay (int, optional): Initial delay between retries in seconds. Defaults to 1.
        """
        auth_manager = oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        self.client = spotipy.Spotify(auth_manager=auth_manager)
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _make_request(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Helper method to make requests with retry logic and exponential backoff.

        Args:
            func (Any): The Spotify client function to call.
            *args (Any): Positional arguments for the Spotify client function.
            **kwargs (Any): Keyword arguments for the Spotify client function.

        Returns:
            Any: The result of the Spotify client function if successful.

        Raises:
            Exception: If the request fails after the maximum number of retries.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except (ConnectionError, HTTPError, ReadTimeout, spotipy.exceptions.SpotifyException) as e:
                LOGGER.warning(f"Attempt {attempt} failed with error: {e}")
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)
                    LOGGER.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    LOGGER.error(f"Failed after {self.max_retries} attempts.")
                    raise

    def get_playlist(self, uri: str) -> Dict[str, Any]:
        """Fetches playlist metadata from Spotify.

        Args:
            uri (str): Spotify URI of the playlist.

        Returns:
            Dict[str, Any]: Metadata for the specified playlist.
        """
        return self._make_request(self.client.playlist, uri)

    def get_audio_features(self, song_id: str) -> Optional[List[Dict[str, Any]]]:
        """Fetches audio features for a song from Spotify.

        Args:
            song_id (str): Spotify ID of the song.

        Returns:
            Optional[List[Dict[str, Any]]]: Audio features for the song, or None if not found.
        """
        return self._make_request(self.client.audio_features, song_id)

    def get_playlist_tracks(self, playlist_uri: str, limit: int = 100, offset: int = 0) -> List[dict]:
        """
        Fetches tracks from a Spotify playlist.
    
        Args:
            playlist_uri (str): Spotify playlist URI.
            limit (int): Maximum number of tracks to fetch per request.
            offset (int): Offset for pagination.
    
        Returns:
            List[dict]: A list of track items from the playlist.
        """
        results = self._make_request(self.client.playlist_items, playlist_uri, limit=limit, offset=offset)
        return results.get('items', [])
    

    # def get_playlist_tracks(self, uri: str) -> List[Dict[str, Any]]:
    #     """Fetches tracks from a Spotify playlist, handling pagination.

    #     Args:
    #         uri (str): Spotify URI of the playlist.

    #     Returns:
    #         List[Dict[str, Any]]: List of tracks in the playlist.
    #     """
    #     tracks = []
    #     results = self.get_playlist(uri)
    #     playlist_tracks = results["tracks"]

    #     while playlist_tracks:
    #         tracks.extend(playlist_tracks["items"])
    #         playlist_tracks = self._make_request(self.client.next, playlist_tracks) if playlist_tracks["next"] else None

    #     return [track["track"] for track in tracks if track.get("track")]

    def get_artist(self, artist_id: str) -> Dict[str, Any]:
        """Fetches artist data from Spotify.

        Args:
            artist_id (str): Spotify ID of the artist.

        Returns:
            Dict[str, Any]: Metadata for the specified artist.
        """
        return self._make_request(self.client.artist, artist_id)

    def search_tracks(self, query: str, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Searches for tracks on Spotify.

        Args:
            query (str): The search query (e.g., "genre:pop").
            limit (int): The maximum number of tracks to return. Defaults to 50.
            offset (int): The offset for paginated results. Defaults to 0.

        Returns:
            Dict[str, Any]: Search results containing tracks.
        """
        return self._make_request(self.client.search, q=query, type="track", limit=limit, offset=offset)

    def get_albums_by_artist(self, artist_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetches albums by an artist.

        Args:
            artist_id (str): Spotify ID of the artist.
            limit (int): The maximum number of albums to fetch. Defaults to 50.

        Returns:
            List[Dict[str, Any]]: List of albums.
        """
        albums = []
        results = self._make_request(self.client.artist_albums, artist_id, limit=limit)
        while results:
            albums.extend(results["items"])
            results = self._make_request(self.client.next, results) if results["next"] else None
        return albums
