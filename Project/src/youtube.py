"""
YouTube URL handling module.
Extracts and validates YouTube video IDs from various URL formats.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse


class YouTubeURLError(Exception):
    """Raised when a YouTube URL is invalid or malformed."""
    pass


# Strict YouTube video ID pattern (exactly 11 characters: alphanumeric, dash, or underscore)
VIDEO_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{11}$")

# Fallback regex patterns matching common YouTube paths
PATH_PATTERNS = [
    re.compile(r"^/embed/([a-zA-Z0-9_-]{11})"),
    re.compile(r"^/shorts/([a-zA-Z0-9_-]{11})"),
    re.compile(r"^/live/([a-zA-Z0-9_-]{11})"),
    re.compile(r"^/v/([a-zA-Z0-9_-]{11})"),
]


def extract_video_id(url: str) -> str:
    """
    Extracts the 11-character YouTube video ID from supported YouTube URL formats.

    Supported formats include:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtube.com/watch?v=VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - https://www.youtube.com/live/VIDEO_ID
    - Raw 11-character video ID string

    Args:
        url: The YouTube URL or raw video ID entered by the user.

    Returns:
        str: Validated 11-character YouTube video ID.

    Raises:
        YouTubeURLError: If the input is empty, invalid, or does not yield a valid 11-character video ID.
    """
    if not url or not isinstance(url, str):
        raise YouTubeURLError("Please enter a YouTube video URL.")

    cleaned_url = url.strip()

    # Case 0: User entered the raw 11-character video ID directly
    if VIDEO_ID_REGEX.match(cleaned_url):
        return cleaned_url

    # Ensure URL has a scheme for proper urlparse handling
    if not cleaned_url.startswith(("http://", "https://")):
        cleaned_url = f"https://{cleaned_url}"

    try:
        parsed = urlparse(cleaned_url)
    except Exception as exc:
        raise YouTubeURLError(f"Invalid URL structure: {exc}") from exc

    hostname = (parsed.netloc or "").lower()

    # Check for valid YouTube domains
    valid_domains = (
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
        "www.youtu.be",
    )

    if not any(hostname == domain or hostname.endswith(f".{domain}") for domain in valid_domains):
        raise YouTubeURLError(
            f"The URL domain '{hostname}' is not a recognized YouTube domain. "
            "Please paste a valid YouTube or youtu.be link."
        )

    video_id: str | None = None

    # Case 1: Shortened youtu.be URL (e.g., https://youtu.be/VIDEO_ID)
    if "youtu.be" in hostname:
        path_parts = parsed.path.strip("/").split("/")
        if path_parts and path_parts[0]:
            candidate = path_parts[0]
            # Strip potential query/anchor artifacts if present in path segment
            candidate = candidate.split("?")[0].split("&")[0]
            if VIDEO_ID_REGEX.match(candidate):
                video_id = candidate

    # Case 2: Standard youtube.com/watch?v=VIDEO_ID
    if not video_id and parsed.path.startswith("/watch"):
        query_params = parse_qs(parsed.query)
        v_param = query_params.get("v")
        if v_param and len(v_param) > 0:
            candidate = v_param[0]
            if VIDEO_ID_REGEX.match(candidate):
                video_id = candidate

    # Case 3: Other paths: /embed/ID, /shorts/ID, /live/ID, /v/ID
    if not video_id:
        for pattern in PATH_PATTERNS:
            match = pattern.match(parsed.path)
            if match:
                video_id = match.group(1)
                break

    # Case 4: General regex search across the URL as an extra safety net
    if not video_id:
        general_match = re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})(?:[&?\/#]|$)", cleaned_url)
        if general_match:
            candidate = general_match.group(1)
            if VIDEO_ID_REGEX.match(candidate):
                video_id = candidate

    # Final validation
    if not video_id or not VIDEO_ID_REGEX.match(video_id):
        raise YouTubeURLError(
            "Could not extract a valid 11-character YouTube video ID. "
            "Please check the URL and try again."
        )

    return video_id


def fetch_video_metadata(video_id: str) -> dict:
    """
    Fetches public video metadata (title, author, thumbnail) using YouTube's free oEmbed API.
    Does NOT require any API keys.

    Args:
        video_id: 11-character YouTube video ID.

    Returns:
        dict: Video metadata containing 'title', 'author', 'thumbnail', and 'url'.
    """
    import json
    import urllib.request

    default_meta = {
        "title": f"YouTube Video ({video_id})",
        "author": "YouTube Creator",
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        "url": f"https://www.youtube.com/watch?v={video_id}",
    }

    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        req = urllib.request.Request(
            oembed_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req, timeout=3.5) as response:
            data = json.loads(response.read().decode("utf-8"))
            return {
                "title": data.get("title", default_meta["title"]),
                "author": data.get("author_name", default_meta["author"]),
                "thumbnail": data.get("thumbnail_url", default_meta["thumbnail"]),
                "url": default_meta["url"],
            }
    except Exception:
        return default_meta

