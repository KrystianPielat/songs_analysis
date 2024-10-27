import os
import csv
import logging
from tqdm.auto import tqdm
from .spotify_manager import SpotifyManager
from .lyrics_manager import LyricsManager
from .audio_features import AudioFeatureExtractor
from .song import Song, SpotifyAudioFeatures
from .youtube_downloader import YouTubeDownloader

LOGGER = logging.getLogger(__name__)

class Playlist:
    def __init__(self, uri, spotify_manager, save_path=None):
        self.uri = uri
        self.save_path = save_path or os.path.join('playlists')
        self.songs = []
        self.spotify_manager = spotify_manager
        self.lyrics_manager = LyricsManager()
        self.youtube_downloader = YouTubeDownloader()
        self.playlist_name = "Unknown"
        self.total_tracks = -1
        self.csv_file = None
        self._initialize_playlist()
        self._populate_songs()

    def _initialize_playlist(self):
        playlist_data = self.spotify_manager.get_playlist(self.uri)
        self.playlist_name = playlist_data.get('name', 'Unknown Playlist')
        self.save_path = os.path.join(self.save_path, self.playlist_name)
        self.csv_file = os.path.join(self.save_path, f"{self.playlist_name}.csv")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.total_tracks = playlist_data['tracks']['total']
        LOGGER.info(f"Found playlist: {self.playlist_name} with {self.total_tracks} tracks.")

    def _populate_songs(self):
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

            song = Song(
                id=track['id'],
                title=track['name'].lower(),
                artist=track['artists'][0]['name'].lower(),
                album_art_url=track['album']['images'][0]['url'] if track['album'].get('images') else None,
                popularity=int(popularity) if popularity else None,
                explicit=track['explicit'],
                album_release_year=int(album_release_year),
                duration_ms=int(duration_ms),
                genres=genres,
                mp3_path=None  # Initialized as None until download
            )
            self.songs.append(song)

    def process_songs(self):
        existing_songs = self._load_existing_songs()
        
        # Use headers from the first song's `to_csv_row()` to ensure all nested fields are included
        if self.songs:
            csv_headers = self.songs[0].to_csv_row().keys()
        
        with open(self.csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=csv_headers)
            if os.path.getsize(self.csv_file) == 0:  # Write header only if CSV is new
                writer.writeheader()

            for song in tqdm(self.songs, desc="Processing Songs", unit="song"):
                # Skip processing if the song already exists in the CSV
                if (song.title, song.artist) in existing_songs:
                    LOGGER.info(f"Skipping {song.title} by {song.artist} as it already exists in CSV.")
                    continue
                self._process_single_song(song, writer)

    def _load_existing_songs(self):
        existing_songs = set()
        if os.path.exists(self.csv_file):
            with open(self.csv_file, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    existing_songs.add((row['title'], row['artist']))
        return existing_songs

    def _process_single_song(self, song, writer):
        self._fetch_spotify_features(song)
        self._fetch_lyrics(song)
        self._download_song(song)

        # Write the song data using the flattened dictionary from to_csv_row
        writer.writerow(song.to_csv_row())
        LOGGER.info(f"Completed processing for song: {song.title} by {song.artist}")

    def _fetch_spotify_features(self, song):
        features = self.spotify_manager.client.audio_features(song.id)
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

    def _fetch_lyrics(self, song):
        song.lyrics = self.lyrics_manager.fetch_lyrics(song.artist, song.title, clean_title=True)
    
    def _download_song(self, song):
        # Construct the base song path without .mp3 extension
        song_path_base = os.path.join(self.save_path, song.title)
        mp3_file_path = os.path.abspath(f"{song_path_base}.mp3")  # Full absolute path with .mp3 extension
        
        # Check if file already exists
        if os.path.exists(mp3_file_path):
            LOGGER.info(f"Skipping download for {song.title} by {song.artist} as it already exists.")
            song.mp3_path = mp3_file_path  # Set the absolute path in the Song instance
            return
        
        search_query = f"{song.artist} - {song.title}"
        best_url = self.youtube_downloader.search_youtube(search_query)
        if best_url:
            success = self.youtube_downloader.download_audio(best_url, song_path_base)  # Pass song_path without .mp3
            if success:
                LOGGER.info(f"Downloaded song: {song.title}")
                song.mp3_path = mp3_file_path  # Save the absolute MP3 path on the song instance
                # Pass self.save_path as the save_directory argument for embedding album art
                self.youtube_downloader.embed_album_art(song, mp3_file_path, self.save_path)
            else:
                LOGGER.warning(f"Failed to download song: {song.title}")