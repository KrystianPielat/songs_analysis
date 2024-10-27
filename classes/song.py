from dataclasses import dataclass, field, asdict
from typing import Optional, List

@dataclass
class SpotifyAudioFeatures:
    danceability: Optional[float] = None
    energy: Optional[float] = None
    key: Optional[int] = None
    loudness: Optional[float] = None
    mode: Optional[int] = None
    speechiness: Optional[float] = None
    acousticness: Optional[float] = None
    instrumentalness: Optional[float] = None
    liveness: Optional[float] = None
    valence: Optional[float] = None
    tempo: Optional[float] = None
    time_signature: Optional[int] = None

@dataclass
class Song:
    id: Optional[str] = None
    title: str = ""
    artist: str = ""
    album_art_url: Optional[str] = None
    popularity: Optional[int] = None
    explicit: Optional[bool] = None
    album_release_year: Optional[int] = None
    duration_ms: Optional[int] = None
    genres: Optional[List[str]] = None
    lyrics: Optional[str] = None
    mp3_path: Optional[str] = None  # Updated attribute to hold the MP3 file path

    audio_features: SpotifyAudioFeatures = field(default_factory=SpotifyAudioFeatures)

    def to_csv_row(self):
        row_data = {}
        for field_name, value in self.__dict__.items():
            if hasattr(value, "__dataclass_fields__"):  # Check if value is a dataclass
                row_data.update(asdict(value))  # Unpack fields from nested dataclass
            else:
                row_data[field_name] = value
        return row_data