import spotipy
import spotipy.oauth2 as oauth2

class SpotifyManager:
    """Class to handle Spotify API interactions."""

    def __init__(self, client_id, client_secret):
        auth_manager = oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        self.client = spotipy.Spotify(auth_manager=auth_manager)

    def get_playlist(self, uri):
        """Fetches playlist metadata from Spotify."""
        return self.client.playlist(uri)

    def get_playlist_tracks(self, uri):
        """Fetches tracks from a Spotify playlist."""
        tracks = []
        playlist = self.get_playlist(uri)
        results = playlist['tracks']
        while results:
            tracks.extend(results['items'])
            results = self.client.next(results) if results['next'] else None
        return [track['track'] for track in tracks if track.get('track') is not None ]
