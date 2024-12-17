import os
import re
import csv
import logging
import random
from typing import List, Optional
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import time
from .spotify_manager import SpotifyManager
from .lyrics_manager import LyricsManager
from .audio_features import AudioFeatureExtractor
from .song import Song, SpotifyAudioFeatures
from .youtube_downloader import YouTubeDownloader
from math import ceil

LOGGER = logging.getLogger(__name__)

def retry_operation(max_retries: int = 5, initial_delay: float = 1.0):
    """
    A decorator for retrying a function upon failure with quadratic backoff.

    Args:
        max_retries (int): Maximum number of retries.
        initial_delay (float): Initial delay (in seconds) before the first retry.

    Returns:
        Callable: The wrapped function with retry logic.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    LOGGER.error(f"Attempt {attempt} failed with error: {e}")
                    if attempt < max_retries:
                        delay = initial_delay * (attempt ** 2)  # Quadratic backoff
                        LOGGER.info(f"Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                    else:
                        raise
        return wrapper
    return decorator

class SongContainer:
    """
    A container for manually managing a collection of songs, fetching their metadata, and downloading them.
    """

    def __init__(self, spotify_manager: SpotifyManager, save_path: Optional[str] = None):
        """
        Initializes the SongContainer instance.

        Args:
            spotify_manager (SpotifyManager): Instance of SpotifyManager for API interactions.
            save_path (Optional[str], optional): Path to save the song data. Defaults to 'songs'.
        """
        self.songs: List[Song] = []
        self.faulty_songs: List[Dict[str, str]] = []
        self.spotify_manager = spotify_manager
        self.lyrics_manager = LyricsManager()
        self.youtube_downloader = YouTubeDownloader()
        self.save_path = os.path.abspath(save_path) if save_path else os.path.abspath('songs')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.csv_file = os.path.join(self.save_path, "songs.csv")
        self.faulty_csv = os.path.join(self.save_path, "faulty_songs.csv")
        self._load_existing_songs()
        self._load_faulty_songs()

    def add_song(self, song: Song) -> bool:
        """
        Adds a song to the songs list if it does not already exist.
        Songs are identified as duplicates by their title and artist (case-insensitive).
    
        Args:
            song (Song): The song object to add.
            
        Returns:
            bool: Whether it was successfully added or not.
        """
        song_key = f"{song.title.lower()}_{song.artist.lower()}"
        existing_keys = {f"{s.title.lower()}_{s.artist.lower()}" for s in self.songs}
        
        if song_key not in existing_keys:
            self.songs.append(song)
            LOGGER.info(f"Added song '{song.title}' by '{song.artist}' to the container.")
            return True
        LOGGER.warning(f"Duplicate song detected: '{song.title}' by '{song.artist}'. Skipping.")
        return False

    def _clean_text(self, text: str, max_length: int = 70) -> str:
        """Cleans the song title by removing invalid characters and truncating if needed.

        Args:
            title (str): Original song title.
            max_length (int, optional): Maximum length of the cleaned title. Defaults to 70.

        Returns:
            str: Cleaned and truncated title.
        """
        text = self.lyrics_manager._clean_title(text)
        text = re.sub(r'[<>:"/\\|?*]', '', text)
        text = text[:max_length] if len(text) > max_length else text
        return text.lower()

    def _load_existing_songs(self):
        """
        Loads existing songs from the CSV file and adds them to the songs list.
        """
        if os.path.exists(self.csv_file):
            with open(self.csv_file, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    audio_features = SpotifyAudioFeatures(
                        danceability=float(row['danceability']) if row['danceability'] else None,
                        energy=float(row['energy']) if row['energy'] else None,
                        key=int(row['key']) if row['key'] else None,
                        loudness=float(row['loudness']) if row['loudness'] else None,
                        mode=int(row['mode']) if row['mode'] else None,
                        speechiness=float(row['speechiness']) if row['speechiness'] else None,
                        acousticness=float(row['acousticness']) if row['acousticness'] else None,
                        instrumentalness=float(row['instrumentalness']) if row['instrumentalness'] else None,
                        liveness=float(row['liveness']) if row['liveness'] else None,
                        valence=float(row['valence']) if row['valence'] else None,
                        tempo=float(row['tempo']) if row['tempo'] else None,
                        time_signature=int(row['time_signature']) if row['time_signature'] else None
                    )

                    song = Song(
                        id=row['id'],
                        title=row['title'],
                        artist=row['artist'],
                        album_art_url=row.get('album_art_url'),
                        popularity=int(row['popularity']) if row['popularity'] else None,
                        explicit=row['explicit'].lower() == 'true',
                        album_release_year=int(row['album_release_year']) if row['album_release_year'] else None,
                        duration_ms=int(row['duration_ms']) if row['duration_ms'] else None,
                        genres=row['genres'].split(',') if row.get('genres') else [],
                        lyrics=row.get('lyrics'),
                        mp3_path=row.get('mp3_path'),
                        csv_path=self.csv_file,
                        audio_features=audio_features
                    )
                    self.add_song(song)
                    LOGGER.debug(f"Loaded existing song '{song.title}' by '{song.artist}' from CSV.")

    def _load_faulty_songs(self):
        """
        Loads faulty songs from the faulty CSV file.
        """
        if os.path.exists(self.faulty_csv):
            with open(self.faulty_csv, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                self.faulty_songs = [row for row in reader]
                LOGGER.info(f"Loaded {len(self.faulty_songs)} faulty songs from CSV.")

    def add_songs_by_genre(self, genre: str, num_tracks: int = 50, offset: int = 0):
        """
        Searches Spotify for songs by genre and adds them to the container.
    
        Args:
            genre (str): The genre to search for.
            num_tracks (int): Number of tracks to fetch (maximum 1000).
            offset (int): Offset for Spotify search results.
        """
        # Check for Spotify's maximum limit
        if num_tracks + offset > 1000:
            raise ValueError("Spotify API limit exceeded: Limit + Offset cannot exceed 1000.")
    
        LOGGER.info(f"Fetching {num_tracks} tracks for genre '{genre}'...")
        tracks = []
        limit = 50
    
        while len(tracks) < num_tracks:
            results = self.spotify_manager.search_tracks(f"genre:{genre}", limit=min(limit, num_tracks - len(tracks)), offset=offset)
            offset += limit
    
            for track in results['tracks']['items']:
                if any(faulty['title'].lower() == track['name'].lower() and faulty['artist'].lower() == track['artists'][0]['name'].lower() for faulty in self.faulty_songs):
                    LOGGER.warning(f"Skipping previously marked faulty song '{track['name']}' by '{track['artists'][0]['name']}'.")
                    continue
    
                song = self._create_song_from_track(track)
                if song:
                    tracks.append(song)

        for track in tracks:
            # genres = eval(track.genres).insert(0, genre)
            # track.genres = str(genres)
            track.genres.insert(0, f"search:{genre}")
            self.add_song(track)
        LOGGER.info(f"Added {len(tracks)} songs for genre '{genre}'.")


    def stratify_tracks_by_genre_and_year(
        self, 
        genre: str, 
        year_buckets: list, 
        num_tracks: int, 
    ):
        """
        Fetches tracks stratified by genre and year buckets, equally distributing the requested
        number of tracks across the year buckets and adding them to the container.

        Args:
            genre (str): The genre to search for.
            year_buckets (list): List of year ranges as tuples, e.g., [(1950, 1960), (1960, 1970)].
            num_tracks (int): Total number of tracks to fetch across all buckets.
        """
        total_buckets = len(year_buckets)
        tracks_per_bucket = ceil(num_tracks / total_buckets)

        LOGGER.info(f"Stratifying {num_tracks} tracks across {total_buckets} year buckets for genre '{genre}'.")
        
        for start_year, end_year in year_buckets:
            LOGGER.info(f"Fetching tracks for genre '{genre}' between {start_year} and {end_year}.")
            
            tracks_added = 0
            offset = 0
            limit = 50  # Spotify API max limit per request

            while tracks_added < tracks_per_bucket:
                # Construct the query
                query = f"genre:{genre} year:{start_year}-{end_year}"
                results = self.spotify_manager.search_tracks(
                    query=query,
                    limit=min(limit, tracks_per_bucket - tracks_added),
                    offset=offset
                )
                offset += limit

                # Process fetched tracks
                for track in results.get('tracks', {}).get('items', []):
                    if any(faulty['title'].lower() == track['name'].lower() and faulty['artist'].lower() == track['artists'][0]['name'].lower() for faulty in self.faulty_songs):
                        LOGGER.warning(f"Skipping previously marked faulty song '{track['name']}' by '{track['artists'][0]['name']}'.")
                        continue

                    song = self._create_song_from_track(track)
                    if song:
                        song.genres.insert(0, f"genre:{genre}")
                    if self.add_song(song):
                        tracks_added += 1
                        LOGGER.info(f"Added song '{song.title}' by '{song.artist}' to the container.")
                    else:
                        LOGGER.warning(f"Failed to add song '{track['name']}' by '{track['artists'][0]['name']}' because its already in the container.")

                    if tracks_added >= tracks_per_bucket:
                        break

                # Stop if no more results
                if not results.get('tracks', {}).get('items', []):
                    LOGGER.warning(f"No more tracks found for genre '{genre}' in {start_year}-{end_year}.")
                    break

        LOGGER.info(f"Stratified tracks fetching completed for genre '{genre}'. Total songs in container: {len(self.songs)}.")

    def add_songs_from_playlist(self, playlist_uri: str):
        """
        Fetches songs from a Spotify playlist and adds them to the container.
    
        Args:
            playlist_uri (str): The Spotify URI of the playlist.
        """
        LOGGER.info(f"Fetching songs from playlist with URI: {playlist_uri}")
        
        # Fetch playlist metadata
        playlist_data = self.spotify_manager.get_playlist(playlist_uri)
        playlist_name = playlist_data.get('name', 'Unknown Playlist')
        total_tracks = playlist_data['tracks']['total']
    
        # Fetch tracks in pages
        offset = 0
        limit = 100  # Spotify allows up to 100 items per request
        while offset < total_tracks:
            tracks_data = self.spotify_manager.get_playlist_tracks(playlist_uri, offset=offset, limit=limit)
            offset += limit
    
            for track_item in tracks_data:
                track = track_item.get('track')
                if not track or track.get('type') != 'track' or not track.get('album'):
                    continue
    
                # Avoid adding already faulty songs
                if any(faulty['title'].lower() == track['name'].lower() and faulty['artist'].lower() == track['artists'][0]['name'].lower() for faulty in self.faulty_songs):
                    LOGGER.warning(f"Skipping previously marked faulty song '{track['name']}' by '{track['artists'][0]['name']}'.")
                    continue
    
                # Create Song object and validate
                song = self._create_song_from_track(track)
                if song:
                    self.add_song(song)
    
        LOGGER.info(f"Added songs from playlist '{playlist_name}' with URI '{playlist_uri}'.")



    @retry_operation(max_retries=5, initial_delay=1.0)
    def _create_song_from_track(self, track) -> Optional[Song]:
        """
        Creates a Song object from a Spotify track.
    
        Args:
            track (dict): A track dictionary from Spotify API.
    
        Returns:
            Optional[Song]: A Song object if valid, otherwise None.
        """
        album = track['album']
        album_release_year = album.get('release_date', '').split('-')[0] if album.get('release_date') else None
        duration_ms = track.get('duration_ms')
        popularity = track.get('popularity') or None
    
        artist_id = track['artists'][0]['id']
        artist_data = self.spotify_manager.get_artist(artist_id)
        genres = artist_data.get('genres', [])
    
        song = Song(
            id=track['id'],
            title=self._clean_text(track['name']),
            artist=self._clean_text(track['artists'][0]['name']),
            album_art_url=album['images'][0]['url'] if album.get('images') else None,
            popularity=int(popularity) if popularity else None,
            explicit=track['explicit'],
            album_release_year=int(album_release_year) if album_release_year else None,
            duration_ms=int(duration_ms),
            genres=genres,
            mp3_path=None,
            csv_path=self.csv_file
        )
    

        return song


    @retry_operation(max_retries=5, initial_delay=1.0)
    def load_audio_features(self, song: Song):
        """
        Loads Spotify audio features for a song.
    
        Args:
            song (Song): The song object for which audio features are to be loaded.
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
            LOGGER.info(f"Loaded audio features for '{song.title}' by '{song.artist}'.")
        else:
            LOGGER.warning(f"Audio features for '{song.title}' by '{song.artist}' could not be loaded.")

    @retry_operation(max_retries=5, initial_delay=1.0)
    def load_lyrics(self, song: Song):
        """
        Loads lyrics for a song.
    
        Args:
            song (Song): The song object for which lyrics are to be loaded.
        """
        song.lyrics = self.lyrics_manager.fetch_lyrics(song.artist, song.title, clean_title=True)
        if song.lyrics:
            LOGGER.info(f"Lyrics for '{song.title}' by '{song.artist}' loaded successfully.")
        else:
            LOGGER.warning(f"Lyrics for '{song.title}' by '{song.artist}' could not be found.")
    
    # @retry_operation(max_retries=5, initial_delay=1.0)
    # def download_song(self, song: Song):
    #     """
    #     Downloads a song from YouTube.
    
    #     Args:
    #         song (Song): The song object to be downloaded.
    #     """
    #     mp3_file_path = os.path.join(self.save_path, f"{song.title}.mp3")
    #     temp_file_path = os.path.join(self.save_path, f"{song.title}.part")
    
    #     # Skip if already downloaded
    #     if os.path.exists(mp3_file_path):
    #         LOGGER.info(f"Skipping download for '{song.title}' by '{song.artist}' as it is already downloaded.")
    #         song.mp3_path = mp3_file_path
    #         return
    
    #     search_query = f"{song.artist} - {song.title}"
    #     best_url = self.youtube_downloader.search_youtube(search_query)
    
    #     if best_url:
    #         try:
    #             # Download the audio file
    #             success = self.youtube_downloader.download_audio(best_url, temp_file_path.rsplit('.part', 1)[0])
    #             if success:
    #                 # Ensure the `.part` file is renamed to the final `.mp3` file
    #                 if os.path.exists(temp_file_path):
    #                     os.rename(temp_file_path, mp3_file_path)
    #                 song.mp3_path = mp3_file_path
    #                 LOGGER.info(f"Downloaded '{song.title}' by '{song.artist}' successfully.")
    
    #                 # Embed album art into the final MP3 file
    #                 self.youtube_downloader.embed_album_art(song, mp3_file_path, self.save_path)
    #             else:
    #                 raise RuntimeError(f"Download failed for '{song.title}' by '{song.artist}'.")
    #         except Exception as e:
    #             LOGGER.error(f"Failed to download '{song.title}' by '{song.artist}' from URL '{best_url}': {e}")
    #             # Clean up incomplete `.part` files
    #             song.mp3_path = None
    #             if os.path.exists(temp_file_path):
    #                 os.remove(temp_file_path)
    #         return
    #     else:
    #         LOGGER.warning(f"No suitable YouTube URL found for '{song.title}' by '{song.artist}'.")
    #         song.mp3_path = None
    @retry_operation(max_retries=5, initial_delay=1.0)
    def download_song(self, song: Song):
        """
        Downloads a song from YouTube.
    
        Args:
            song (Song): The song object to be downloaded.
        """
        mp3_file_path = os.path.join(self.save_path, f"{song.title}.mp3")
        temp_file_path = os.path.join(self.save_path, f"{song.title}.part")
    
        # Skip if already downloaded
        if os.path.exists(mp3_file_path):
            LOGGER.info(f"Skipping download for '{song.title}' by '{song.artist}' as it is already downloaded.")
            song.mp3_path = mp3_file_path
            return
    
        search_query = f"{song.artist} - {song.title}"
        try:
            best_url = self.youtube_downloader.search_youtube(search_query)
            if not best_url:
                LOGGER.warning(f"No suitable YouTube URL found for '{song.title}' by '{song.artist}'.")
                song.mp3_path = None
                return
    
            # Download the audio file
            success = self.youtube_downloader.download_audio(best_url, temp_file_path.rsplit('.part', 1)[0])
            if success:
                # Ensure the `.part` file is renamed to the final `.mp3` file
                if os.path.exists(temp_file_path):
                    os.rename(temp_file_path, mp3_file_path)
                song.mp3_path = mp3_file_path
                LOGGER.info(f"Downloaded '{song.title}' by '{song.artist}' successfully.")
    
                # Embed album art into the final MP3 file
                self.youtube_downloader.embed_album_art(song, mp3_file_path, self.save_path)
            else:
                raise RuntimeError(f"Download failed for '{song.title}' by '{song.artist}'.")
    
        except Exception as e:
            # Detect CAPTCHA-specific error
            if "Sign in to confirm you’re not a bot" in str(e):
                LOGGER.error(f"CAPTCHA error encountered for '{song.title}' by '{song.artist}': {e}")
                raise CaptchaException("CAPTCHA error: unable to search YouTube.") from e
    
            LOGGER.error(f"Failed to download '{song.title}' by '{song.artist}' from URL '{best_url}': {e}")
    
            # Clean up incomplete `.part` files
            song.mp3_path = None
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)


    def fetch_metadata(self):
        """
        Fetches Spotify audio features and lyrics for all songs that are missing either.
        """
        LOGGER.info("Fetching metadata for all songs...")
        
        # Filter the songs that need metadata fetching
        songs_to_fetch = [song for song in self.songs if not song.lyrics or not song.audio_features.danceability]
    
        if not songs_to_fetch:
            LOGGER.info("All songs already have complete metadata. Nothing to fetch.")
            return
    
        with tqdm(total=len(songs_to_fetch), desc="Fetching metadata", unit="song") as pbar:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(self._fetch_metadata_for_song, song): song for song in songs_to_fetch}
                for future in as_completed(futures):
                    song = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        LOGGER.error(f"Error fetching metadata for '{song.title}' by '{song.artist}': {e}")
                    finally:
                        pbar.update(1)


    
    def _fetch_metadata_for_song(self, song: Song):
        """
        Fetches Spotify audio features and lyrics for a single song.
    
        Args:
            song (Song): The song for which metadata is fetched.
        """
        self.load_audio_features(song)
        self.load_lyrics(song)

    # def download_songs(self):
    #     """
    #     Downloads all songs in the container using YouTube.
    #     """
    #     LOGGER.info("Downloading songs...")
    #     songs_to_download = [ song for song in self.songs if not song.mp3_path or not os.path.exists(os.path.join(self.save_path, song.mp3_path)) ]
    #     with tqdm(total=len(songs_to_download), desc="Downloading songs", unit="song") as pbar:
    #         with ThreadPoolExecutor(max_workers=8) as executor:
    #             futures = {
    #                 executor.submit(self.download_song, song): song
    #                 for song in songs_to_download
    #             }
    #             for future in as_completed(futures):
    #                 song = futures[future]
    #                 try:
    #                     future.result()
    #                 except Exception as e:
    #                     LOGGER.error(f"Error downloading '{song.title}' by '{song.artist}': {e}")
    #                 finally:
    #                     pbar.update(1)

    def download_songs(self):
        """
        Downloads all songs in the container using YouTube.
        Stops execution if a CAPTCHA error is encountered.
        """
        LOGGER.info("Starting to download songs...")
        songs_to_download = [
            song for song in self.songs
            if not song.mp3_path or not os.path.exists(os.path.join(self.save_path, song.mp3_path))
        ]
        
        try:
            with tqdm(total=len(songs_to_download), desc="Downloading songs", unit="song") as pbar:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    futures = {
                        executor.submit(self.download_song, song): song
                        for song in songs_to_download
                    }
                    for future in as_completed(futures):
                        song = futures[future]
                        try:
                            future.result()
                        except RuntimeError as e:
                            LOGGER.error(f"Error downloading '{song.title}' by '{song.artist}': {e}")
                        except CaptchaException as e: 
                            LOGGER.error("CAPTCHA error detected. Stopping all downloads.")
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise CaptchaException("CAPTCHA error: Download process stopped.") from e
                        finally:
                            pbar.update(1)
        except RuntimeError as e:
            LOGGER.error(f"Download process stopped due to: {e}")
        except Exception as e:
            LOGGER.error(f"Unexpected error during downloads: {e}")
        except CaptchaException as e: 
            LOGGER.error(f"Download process stopped due captcha")
        else:
            LOGGER.info("All downloads completed successfully.")
    
        

    def _is_song_valid(self, song: Song) -> bool:
        """Validates that a song has no missing critical data."""
        return all([
            song.id, song.title, song.artist, song.duration_ms,
            song.audio_features.danceability is not None, song.lyrics,
            song.audio_features.energy is not None
        ])

    def clean_faulty(self):
        """
        Removes faulty songs from the songs list and appends them to the faulty CSV file.
        Ensures no duplicates in the faulty CSV file.
        """
        LOGGER.info("Cleaning faulty songs...")
    
        # Load existing faulty entries
        existing_faulty = set()
        if os.path.exists(self.faulty_csv):
            with open(self.faulty_csv, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                existing_faulty = {row['title'].lower() + row['artist'].lower() for row in reader}
    
        # Identify faulty songs
        faulty_entries = []
        valid_songs = []
        for song in self.songs:
            if not self._is_song_valid(song):
                song_key = song.title.lower() + song.artist.lower()
                if song_key not in existing_faulty:
                    faulty_entries.append({'title': song.title, 'artist': song.artist})
                    existing_faulty.add(song_key)
            else:
                valid_songs.append(song)
    
        # Update the songs list with valid songs only
        self.songs = valid_songs
    
        # Append new faulty entries to the CSV
        if faulty_entries:
            with open(self.faulty_csv, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['title', 'artist'])
                if os.path.getsize(self.faulty_csv) == 0:  # Write header if file is new
                    writer.writeheader()
                writer.writerows(faulty_entries)
            LOGGER.info(f"Appended {len(faulty_entries)} faulty songs to {self.faulty_csv}.")
    
        LOGGER.info(f"Removed {len(faulty_entries)} faulty songs from the container.")


    # def save_songs_to_csv(self):
    #     """
    #     Saves the current list of songs to the CSV file.
    #     Appends to the file if it already exists, avoiding duplicates.
    #     """
    #     LOGGER.info("Saving songs to CSV...")
    #     if not self.songs:
    #         LOGGER.warning("No songs to save.")
    #         return
    
    #     existing_titles = set()
    #     if os.path.exists(self.csv_file):
    #         with open(self.csv_file, mode='r', encoding='utf-8') as file:
    #             reader = csv.DictReader(file)
    #             existing_titles = {row['title'].lower() + row['artist'].lower() for row in reader}
    
    #     with open(self.csv_file, mode='a', newline='', encoding='utf-8') as file:
    #         writer = csv.DictWriter(file, fieldnames=self.songs[0].to_csv_row().keys())
    #         if os.path.getsize(self.csv_file) == 0:  # Write header if file is new
    #             writer.writeheader()
    
    #         for song in self.songs:
    #             song_key = song.title.lower() + song.artist.lower()
    #             if song_key not in existing_titles:
    #                 writer.writerow(song.to_csv_row())
    #                 existing_titles.add(song_key)
    #                 LOGGER.info(f"Saved song '{song.title}' by '{song.artist}' to CSV.")
    
    #     LOGGER.info("Finished saving songs to CSV.")

    def save_songs_to_csv(self):
        """
        Saves the current list of songs to the CSV file.
        Appends to the file if it already exists, avoiding duplicates.
        """
        LOGGER.info("Saving songs to CSV...")
        if not self.songs:
            LOGGER.warning("No songs to save.")
            return
    
        with open(self.csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self.songs[0].to_csv_row().keys())
            if os.path.getsize(self.csv_file) == 0:  # Write header if file is new
                writer.writeheader()
    
            for song in self.songs:
                writer.writerow(song.to_csv_row())
    
        LOGGER.info("Finished saving songs to CSV.")