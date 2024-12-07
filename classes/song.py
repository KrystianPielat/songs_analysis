from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

@dataclass
class SpotifyAudioFeatures:
    """Dataclass representing Spotify's audio features for a song."""

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
    """Dataclass representing a song with various metadata and Spotify audio features."""

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
    mp3_path: Optional[str] = None
    csv_path: Optional[str] = None

    audio_features: SpotifyAudioFeatures = field(default_factory=SpotifyAudioFeatures)

    def to_csv_row(self) -> Dict[str, Any]:
        """Converts the song and its audio features to a dictionary suitable for CSV writing.

        Returns:
            Dict[str, Any]: A dictionary representation of the song, with nested audio features expanded.
        """
        row_data = {}
        for field_name, value in self.__dict__.items():
            if hasattr(value, "__dataclass_fields__"):  # Check if value is a dataclass
                row_data.update(asdict(value))  # Unpack fields from nested dataclass
            else:
                row_data[field_name] = value
        return row_data