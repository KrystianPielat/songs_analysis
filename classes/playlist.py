import os
import csv
import logging
import re
from typing import Optional, List, Dict, Tuple, Any, Literal
from tqdm.auto import tqdm
from .spotify_manager import SpotifyManager
from .lyrics_manager import LyricsManager
from .audio_features import AudioFeatureExtractor
from .song import Song, SpotifyAudioFeatures
from .youtube_downloader import YouTubeDownloader

LOGGER = logging.getLogger(__name__)

class Playlist:
    """Class for managing a playlist of songs, fetching song data from Spotify, lyrics from various sources,
    and downloading audio files from YouTube."""

    def __init__(self, uri: str, spotify_manager: SpotifyManager, save_path: Optional[str] = None) -> None:
        """
        Initializes a Playlist instance with URI, Spotify manager, and optional save path.

        Args:
            uri (str): Spotify playlist URI.
            spotify_manager (SpotifyManager): Instance of SpotifyManager for API interactions.
            save_path (Optional[str], optional): Path to save the playlist data. Defaults to 'playlists'.
        """
        self.uri = uri
        self.save_path = os.path.abspath(save_path) if save_path else os.path.abspath('playlists')
        self.songs = []
        self.spotify_manager = spotify_manager
        self.lyrics_manager = LyricsManager()
        self.youtube_downloader = YouTubeDownloader()
        self.playlist_name = "Unknown"
        self.total_tracks = -1
        self.csv_file = None
        self._initialize_playlist()
        self._populate_songs()

    def _initialize_playlist(self) -> None:
        """Fetches playlist metadata from Spotify and sets up storage paths."""

        playlist_data = self.spotify_manager.get_playlist(self.uri)
        self.playlist_name = playlist_data.get('name', 'Unknown Playlist')
        self.save_path = os.path.join(self.save_path, self.playlist_name)
        self.csv_file = os.path.abspath(os.path.join(self.save_path, f"{self.playlist_name}.csv"))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.total_tracks = playlist_data['tracks']['total']
        LOGGER.info(f"Found playlist: {self.playlist_name} with {self.total_tracks} tracks.")

    def _populate_songs(self) -> None:
        """Populates the songs list with metadata from Spotify for each track in the playlist."""

        tracks = self.spotify_manager.get_playlist_tracks(self.uri)
        for track in tqdm(tracks, desc='Fetching songs info', unit='song'):
            if track.get('type') != 'track' or track.get('album') is None:
                continue

            album_release_year = track['album'].get('release_date', '').split('-')[0]
            duration_ms = track.get('duration_ms')
            popularity = track.get('popularity') or None
            artist_id = track['artists'][0]['id']
            artist_data = self.spotify_manager.get_artist(artist_id)
            genres = artist_data.get('genres', [])

            # Clean the title and artist upon initialization
            song = Song(
                id=track['id'],
                title=self._clean_title(track['name']),
                artist=self._clean_title(track['artists'][0]['name']),
                album_art_url=track['album']['images'][0]['url'] if track['album'].get('images') else None,
                popularity=int(popularity) if popularity else None,
                explicit=track['explicit'],
                album_release_year=int(album_release_year),
                duration_ms=int(duration_ms),
                genres=genres,
                mp3_path=None  # Initialized as None until download
            )
            self.songs.append(song)

    def _clean_title(self, title: str, max_length: int = 70) -> str:
        """Cleans the song title by removing invalid characters and truncating if needed.

        Args:
            title (str): Original song title.
            max_length (int, optional): Maximum length of the cleaned title. Defaults to 70.

        Returns:
            str: Cleaned and truncated title.
        """
        title = self.lyrics_manager._clean_title(title)
        title = re.sub(r'[<>:"/\\|?*]', '', title)
        title = title[:max_length] if len(title) > max_length else title
        return title.lower()

    def _generate_unique_mp3_path(self, title: str) -> str:
        """Generates a unique MP3 file path for a given song title within the save directory.

        Args:
            title (str): The cleaned song title.

        Returns:
            str: Absolute path for the unique MP3 file.
        """
        song_path_base = os.path.join(self.save_path, title)
        mp3_file_path = os.path.abspath(f"{song_path_base}.mp3")
        counter = 1
        while os.path.exists(mp3_file_path):
            mp3_file_path = os.path.abspath(f"{song_path_base}_{counter}.mp3")
            counter += 1
        return mp3_file_path

    def process_songs(self) -> None:
        """Processes each song by fetching or downloading missing data, then saves to CSV."""
        existing_songs = self._load_existing_songs()
        
        # Use headers from the first song's `to_csv_row()` to ensure all nested fields are included
        if self.songs:
            csv_headers = self.songs[0].to_csv_row().keys()

        with open(self.csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=csv_headers)
            if os.path.getsize(self.csv_file) == 0:  # Write header only if CSV is new
                writer.writeheader()

            for song in tqdm(self.songs, desc="Processing Songs", unit="song"):
                # Check if the song already exists in the CSV
                if (song.title, song.artist) in existing_songs:
                    # Use existing mp3_path if set and exists
                    if existing_songs[(song.title, song.artist)] is not None and os.path.exists(existing_songs[(song.title, song.artist)]):
                        LOGGER.info(f"Skipping {song.title} by {song.artist} as it already exists with MP3 file.")
                        continue
                    else:
                        LOGGER.info(f"CSV entry exists for {song.title} by {song.artist}, but MP3 file is missing. Downloading...")
                        song.mp3_path = self._generate_unique_mp3_path(song.title)  # Set new path for missing file
                        self._download_song(song)
                else:
                    # Process and download new song if it's not in CSV
                    self._process_single_song(song, writer)

    def _load_existing_songs(self) -> Dict[Tuple[str, str], Optional[str]]:
        """Loads existing songs from the CSV to avoid reprocessing.

        Returns:
            Dict[Tuple[str, str], Optional[str]]: Dictionary mapping (title, artist) to mp3_path.
        """
        existing_songs = {}
        if os.path.exists(self.csv_file):
            with open(self.csv_file, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    existing_songs[(row['title'], row['artist'])] = row.get('mp3_path')
        return existing_songs

    def _process_single_song(self, song: Song, writer: csv.DictWriter) -> None:
        """Processes a single song by fetching Spotify features, lyrics, and downloading audio.

        Args:
            song (Song): Song object containing song metadata.
            writer (csv.DictWriter): CSV writer to save song data to file.
        """
        self._fetch_spotify_features(song)
        self._fetch_lyrics(song)
        self._download_song(song)

        # Write the song data using the flattened dictionary from to_csv_row
        writer.writerow(song.to_csv_row())
        LOGGER.info(f"Completed processing for song: {song.title} by {song.artist}")

    def _fetch_spotify_features(self, song: Song) -> None:
        """Fetches Spotify audio features and assigns them to the Song object.

        Args:
            song (Song): Song object to which features will be assigned.
        """
        features = self.spotify_manager.get_audio_features(song.id)
        if features:
            features = features[0]
            song.audio_features = SpotifyAudioFeatures(
                danceability=features.get('danceability'),
                energy=features.get('energy'),
                key=features.get('key'),
                loudness=features.get('loudness'),
                mode=features.get('mode'),
                speechiness=features.get('speechiness'),
                acousticness=features.get('acousticness'),
                instrumentalness=features.get('instrumentalness'),
                liveness=features.get('liveness'),
                valence=features.get('valence'),
                tempo=features.get('tempo'),
                time_signature=features.get('time_signature')
            )
            LOGGER.info(f"Fetched Spotify audio features for {song.title} by {song.artist}")

    def _fetch_lyrics(self, song: Song) -> None:
        """Fetches lyrics for a song using the LyricsManager.

        Args:
            song (Song): Song object to which lyrics will be assigned.
        """
        song.lyrics = self.lyrics_manager.fetch_lyrics(song.artist, song.title, clean_title=True)

    def _download_song(self, song: Song) -> None:
        """Downloads the song audio from YouTube and embeds album art.

        Args:
            song (Song): Song object for which the audio will be downloaded.
        """
        # Generate unique file path using the cleaned title
        mp3_file_path = self._generate_unique_mp3_path(song.title)
        
        # Skip download if file already exists
        if os.path.exists(mp3_file_path):
            LOGGER.info(f"Skipping download for {song.title} by {song.artist} as it already exists.")
            song.mp3_path = mp3_file_path  # Set the absolute path in the Song instance
            return
        
        search_query = f"{song.artist} - {song.title}"
        best_url = self.youtube_downloader.search_youtube(search_query)
        if best_url:
            success = self.youtube_downloader.download_audio(best_url, mp3_file_path.rsplit('.mp3', 1)[0])  # Download without .mp3
            if success:
                LOGGER.info(f"Downloaded song: {song.title}")
                song.mp3_path = mp3_file_path  # Store as absolute path
                self.youtube_downloader.embed_album_art(song, mp3_file_path, self.save_path)  # Embed album art
            else:
                LOGGER.warning(f"Failed to download song: {song.title}")
