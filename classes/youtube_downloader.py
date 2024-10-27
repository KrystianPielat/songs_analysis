import os
import urllib.request
import yt_dlp
import logging
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, error
from typing import Optional

LOGGER = logging.getLogger(__name__)

class YouTubeDownloader:
    def __init__(self):
        pass

    def search_youtube(self, query: str) -> Optional[str]:
        """Search for a YouTube video and return the best URL match."""
        try:
            from youtube_search import YoutubeSearch
            results = YoutubeSearch(query, max_results=1).to_dict()
            url = f"https://www.youtube.com{results[0]['url_suffix']}"
            LOGGER.info(f"Found YouTube URL for '{query}': {url}")
            return url
        except IndexError:
            LOGGER.warning(f"No valid YouTube URL found for '{query}'")
            return None

    def download_audio(self, url: str, output_path: str) -> bool:
        """Download audio from YouTube URL."""
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'quiet': True,
            'postprocessors': [
                {
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                },
                {
                    'key': 'FFmpegMetadata',
                }
            ]
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            LOGGER.info(f"Downloaded audio to '{output_path}'")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to download from YouTube URL '{url}': {e}")
            return False

    def embed_album_art(self, song, song_path: str, save_directory: str) -> None:
        """Embed album art into the downloaded MP3 file."""
        if song.album_art_url:
            album_art_path = os.path.join(save_directory, f"{song.title}.jpg")
            urllib.request.urlretrieve(song.album_art_url, album_art_path)
            try:
                audio = MP3(song_path, ID3=ID3)
                audio.add_tags()
            except error:
                pass  # Tags already exist, so we skip adding

            with open(album_art_path, 'rb') as img:
                audio.tags.add(
                    APIC(
                        encoding=3,  # UTF-8
                        mime='image/jpeg',  # JPEG image
                        type=3,  # Cover image
                        desc='Cover',
                        data=img.read()
                    )
                )
            audio.save()
            os.remove(album_art_path)
            LOGGER.info(f"Embedded album art for {song.title} by {song.artist}")
        else:
            LOGGER.warning(f"No album art URL for {song.title} by {song.artist}")
