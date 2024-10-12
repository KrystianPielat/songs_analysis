import os
import csv
import logging
from tqdm.auto import tqdm
from .song import Song
from .audio_features import AudioFeatureExtractor
from .spotify_manager import SpotifyManager
from classes.log import setup_logging

LOGGER = logging.getLogger(__name__)


class Playlist:
    def __init__(self, uri, spotify_manager, save_path=None):
        self.uri = uri
        self.save_path = save_path
        self.songs = []
        self.spotify_manager = spotify_manager
        self.playlist_name = "Unknown"
        self.total_tracks = -1
        self.csv_file = None
        self._initialize_playlist()
        self._populate_songs()

    def _initialize_playlist(self):
        """Initializes the Spotify playlist details."""
        LOGGER.debug(f"Initializing playlist for URI: {self.uri}")
        playlist_data = self.spotify_manager.get_playlist(self.uri)
        self.playlist_name = playlist_data.get('name', 'Unknown Playlist')
        self.save_path = self.save_path or os.path.join(os.path.dirname(os.path.realpath('__file__')), 'playlists', self.playlist_name)
        self.csv_file = os.path.join(self.save_path, f"{self.playlist_name}.csv")
        LOGGER.debug(f"Save path set to: {self.save_path}, CSV file path: {self.csv_file}")

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            LOGGER.debug(f"Directory created: {self.save_path}")
        self.total_tracks = playlist_data['tracks']['total']
        LOGGER.info(f"Found playlist: {self.playlist_name} with {self.total_tracks} tracks.")

    def _populate_songs(self):
        """Fetches songs from the Spotify playlist."""
        LOGGER.debug("Fetching songs from the playlist.")
        tracks = self.spotify_manager.get_playlist_tracks(self.uri)
        
        for track in tracks:
            # Extract album release year
            album_release_year = Song._extract_album_release_year(track['album'])

            # Get song duration in milliseconds
            duration_ms = track.get('duration_ms', None)

            # Get the popularity and explicitly set it to None if it's 0 or not found
            popularity = track.get('popularity')
            if popularity == 0 or popularity is None:
                popularity = None

            song = Song(
                id=track.get('id', None),
                title=track.get('name', 'Unknown Title'),
                artist=track['artists'][0].get('name', 'Unknown Artist'),
                album_art_url=track['album']['images'][0]['url'] if track['album'].get('images') else None,
                popularity=popularity,
                explicit=track.get('explicit', None),
                album_release_year=album_release_year,  # Set the release year
                duration_ms=duration_ms,  # Set the duration in milliseconds
                save_path=self.save_path
            )
            LOGGER.debug(f"Adding song: {song.title} by {song.artist}")
            song.fetch_spotify_features(self.spotify_manager.client)
            self.songs.append(song)
        LOGGER.debug(f"Total songs added: {len(self.songs)}")

    def load_songs(self, download_when_no_lyrics=False):
        """Downloads all songs in the playlist, saves to CSV, and extracts audio features."""
        LOGGER.debug("Loading songs to download and extract features.")
        existing_songs = set()
    
        # Read existing songs from the CSV file (if it exists)
        if os.path.exists(self.csv_file):
            LOGGER.debug(f"CSV file exists. Reading existing songs from: {self.csv_file}")
            with open(self.csv_file, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                # Add songs by title to existing_songs set
                for row in reader:
                    existing_songs.add(row['title'])
            LOGGER.debug(f"Existing songs found: {len(existing_songs)}")
    
        with open(self.csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
    
            # Write the header if it's a new CSV
            if len(existing_songs) == 0:
                LOGGER.debug("No existing songs found, writing CSV header.")
                writer.writerow(
                    ['title', 'artist', 'mp3_path', 'lyrics', 'popularity', 'explicit', 'album_release_year', 'duration_ms'] +  # Include album release year and duration
                    ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                     'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'] +
                    [f'mfcc_{i+1}' for i in range(13)] + 
                    [f'chroma_{i+1}' for i in range(12)] +
                    [f'spectral_contrast_{i+1}' for i in range(7)] + 
                    ['tempo_extracted', 'zcr']
                )
                file.flush()  # Ensure header is written immediately
                LOGGER.debug("Header written to CSV.")
    
            with tqdm(total=len(self.songs), desc="Processing Songs", leave=True, unit='song') as progress_bar:
                for song in self.songs:
                    try:
                        # Skip processing if the song is already in the CSV (i.e., in existing_songs set)
                        if song.title in existing_songs:
                            LOGGER.info(f"Skipping song {song.title}, already exists in CSV.")
                            progress_bar.update(1)
                            continue
    
                        LOGGER.info(f"Processing song: {song.title}")
                        song_path = os.path.join(self.save_path, f"{song.title}.mp3")
                        absolute_song_path = os.path.abspath(song_path)
    
                        # Check if the song file already exists
                        if os.path.exists(absolute_song_path):
                            LOGGER.info(f"Song {song.title} already downloaded, skipping download.")
                        else:
                            LOGGER.info(f"Downloading song: {song.title}")
                            song.download_song()
    
                        # Fetch lyrics
                        lyrics = song.fetch_lyrics()
                        LOGGER.debug(f"Lyrics for {song.title}: {lyrics}")
    
                        # Extract audio features
                        audio_extractor = AudioFeatureExtractor()
                        extracted_audio_features = audio_extractor.extract_audio_features(absolute_song_path)
                        LOGGER.debug(f"Extracted audio features for {song.title}: {extracted_audio_features}")
    
                        # Check if data is valid before writing to CSV
                        if extracted_audio_features is None or extracted_audio_features.size == 0:
                            LOGGER.error(f"Audio features for {song.title} are empty. Skipping.")
                            progress_bar.update(1)
                            continue
    
                        LOGGER.debug(f"Preparing to write data for song: {song.title}")
                        LOGGER.debug(f"Row data: {[song.title, song.artist, absolute_song_path, lyrics or 'No lyrics found']}")

                        # Write the song information and relevant Spotify + MP3 audio features to the CSV
                        writer.writerow([
                            song.title, song.artist, absolute_song_path, lyrics or "No lyrics found",
                            song.popularity, song.explicit, song.album_release_year, song.duration_ms  # Add release year and duration to CSV
                        ] + [
                            song.relevant_audio_features.get('danceability'),
                            song.relevant_audio_features.get('energy'),
                            song.relevant_audio_features.get('key'),
                            song.relevant_audio_features.get('loudness'),
                            song.relevant_audio_features.get('mode'),
                            song.relevant_audio_features.get('speechiness'),
                            song.relevant_audio_features.get('acousticness'),
                            song.relevant_audio_features.get('instrumentalness'),
                            song.relevant_audio_features.get('liveness'),
                            song.relevant_audio_features.get('valence'),
                            song.relevant_audio_features.get('tempo'),
                            song.relevant_audio_features.get('time_signature')
                        ] + list(extracted_audio_features))
    
                        LOGGER.debug(f"Data written for song: {song.title}")
                        file.flush()  # Ensure row data is written immediately
                        progress_bar.update(1)
                        LOGGER.info(f"Successfully processed and saved song: {song.title}")
    
                    except KeyboardInterrupt:
                        LOGGER.error("Process interrupted by user (KeyboardInterrupt). Exiting gracefully.")
                        raise
    
                    except Exception as e:
                        LOGGER.error(f"There was an error processing song: {song.title} by {song.artist}: {e}")
                        progress_bar.update(1)
    
        LOGGER.info(f"Audio features and metadata saved to {self.csv_file}")
