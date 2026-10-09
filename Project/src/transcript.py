"""
Transcript retrieval and processing module.
Fetches YouTube subtitles using youtube-transcript-api and cleans the transcript text.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

try:
    from youtube_transcript_api._errors import (
        IpBlocked,
        RequestBlocked,
        YouTubeRequestFailed,
    )
except ImportError:
    # Older or alternative versions of youtube-transcript-api
    IpBlocked = Exception  # type: ignore
    RequestBlocked = Exception  # type: ignore
    YouTubeRequestFailed = Exception  # type: ignore


class TranscriptError(Exception):
    """Base exception for transcript retrieval failures."""
    pass


class TranscriptNotAvailableError(TranscriptError):
    """Raised when transcripts are disabled or not available for a video."""
    pass


class VideoNotAvailableError(TranscriptError):
    """Raised when the video itself is unavailable."""
    pass


class TranscriptBlockedError(TranscriptError):
    """Raised when YouTube blocks the transcript request (rate-limiting or IP restriction)."""
    pass


def _get_api_instance() -> YouTubeTranscriptApi:
    """Instantiates YouTubeTranscriptApi safely across different package versions."""
    try:
        return YouTubeTranscriptApi()
    except TypeError:
        # If static class in older versions
        return YouTubeTranscriptApi  # type: ignore


def fetch_transcript(
    video_id: str,
    preferred_languages: Optional[List[str]] = None
) -> Tuple[str, dict]:
    """
    Fetches the transcript for a YouTube video.
    Prefers manually created captions over auto-generated captions,
    prioritizing English languages ('en', 'en-US', 'en-GB').

    Args:
        video_id: The validated 11-character YouTube video ID.
        preferred_languages: List of language codes in order of preference.
            Defaults to ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU'].

    Returns:
        Tuple[str, dict]:
            - Cleaned transcript text.
            - Metadata dictionary (language, is_generated, snippet_count, etc.).

    Raises:
        TranscriptNotAvailableError: Video has no subtitles or transcripts are disabled.
        VideoNotAvailableError: Video cannot be reached or is private.
        TranscriptBlockedError: YouTube rate-limiting or blocking requests.
        TranscriptError: Any other unexpected retrieval error.
    """
    if preferred_languages is None:
        preferred_languages = ["en", "en-US", "en-GB", "en-CA", "en-AU"]

    api = _get_api_instance()
    transcript_list = None
    selected_transcript = None

    try:
        # Retrieve the TranscriptList for the video
        if hasattr(api, "list"):
            transcript_list = api.list(video_id)
        elif hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        else:
            raise TranscriptError("Unsupported version of youtube-transcript-api: missing list method.")

    except TranscriptsDisabled as exc:
        raise TranscriptNotAvailableError(
            "Subtitles and transcripts are disabled for this video."
        ) from exc
    except VideoUnavailable as exc:
        raise VideoNotAvailableError(
            "This video is unavailable (it may be private, age-restricted, or removed)."
        ) from exc
    except (IpBlocked, RequestBlocked) as exc:
        raise TranscriptBlockedError(
            "YouTube has temporarily restricted caption requests from this IP. "
            "Please try again in a few moments."
        ) from exc
    except YouTubeRequestFailed as exc:
        raise TranscriptError(
            f"Network error while contacting YouTube: {exc}"
        ) from exc
    except Exception as exc:
        err_msg = str(exc)
        if "Subtitles are disabled" in err_msg or "disabled" in err_msg.lower():
            raise TranscriptNotAvailableError(
                "Subtitles and transcripts are disabled for this video."
            ) from exc
        raise TranscriptError(f"Could not retrieve transcript list: {err_msg}") from exc

    # Attempt selection strategies:
    # 1. Manually created in preferred languages (English)
    try:
        selected_transcript = transcript_list.find_manually_created_transcript(preferred_languages)
    except Exception:
        pass

    # 2. Automatically generated in preferred languages
    if selected_transcript is None:
        try:
            selected_transcript = transcript_list.find_generated_transcript(preferred_languages)
        except Exception:
            pass

    # 3. Any transcript matching preferred languages
    if selected_transcript is None:
        try:
            selected_transcript = transcript_list.find_transcript(preferred_languages)
        except Exception:
            pass

    # 4. Fallback to any available transcript if English was not found
    if selected_transcript is None:
        try:
            available = list(transcript_list)
            if available:
                selected_transcript = available[0]
        except Exception:
            pass

    if selected_transcript is None:
        raise TranscriptNotAvailableError(
            "Sorry, this video does not have an accessible transcript. "
            "Please make sure the video has captions/subtitles enabled."
        )

    # Fetch the actual transcript content
    try:
        fetched_data = selected_transcript.fetch()
    except Exception as exc:
        raise TranscriptError(f"Failed to download caption content: {exc}") from exc

    # Extract raw text snippets and timestamps
    snippets = []
    timestamped_snippets = []
    if hasattr(fetched_data, "__iter__"):
        for item in fetched_data:
            text_val = ""
            start_val = 0.0
            duration_val = 0.0

            if hasattr(item, "text"):
                text_val = item.text
                start_val = getattr(item, "start", 0.0)
                duration_val = getattr(item, "duration", 0.0)
            elif isinstance(item, dict):
                text_val = item.get("text", "")
                start_val = item.get("start", 0.0)
                duration_val = item.get("duration", 0.0)
            elif isinstance(item, str):
                text_val = item

            if text_val:
                snippets.append(text_val)
                timestamped_snippets.append({
                    "text": text_val,
                    "start": float(start_val),
                    "duration": float(duration_val),
                })

    if not snippets:
        raise TranscriptNotAvailableError(
            "The retrieved transcript was empty. No readable subtitles were found."
        )

    raw_text = " ".join(snippets)
    cleaned_text = clean_transcript(raw_text)

    total_duration = 0.0
    if timestamped_snippets:
        last = timestamped_snippets[-1]
        total_duration = last["start"] + last["duration"]

    metadata = {
        "language": getattr(selected_transcript, "language", "Unknown"),
        "language_code": getattr(selected_transcript, "language_code", "Unknown"),
        "is_generated": getattr(selected_transcript, "is_generated", False),
        "snippet_count": len(snippets),
        "snippets": timestamped_snippets,
        "duration_seconds": total_duration,
    }

    return cleaned_text, metadata


def clean_transcript(text: str) -> str:
    """
    Cleans transcript text by:
    - Stripping bracketed stage directions/cues (e.g., [Music], [Applause], [laughter])
    - Collapsing redundant whitespace, tabs, and newlines
    - Preserving proper sentence punctuation
    - Trimming leading and trailing whitespace

    Args:
        text: Raw combined transcript string.

    Returns:
        str: Cleaned, readable text.
    """
    if not text:
        return ""

    # Remove caption cues like [Music], [Applause], [Laughter], [Cheering], etc.
    cleaned = re.sub(r"\[(Music|Applause|Laughter|Cheering|Audio|Inaudible|Silence|Music & Sound)\]", " ", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\((Music|Applause|Laughter|Cheering)\)", " ", cleaned, flags=re.IGNORECASE)

    # Replace newlines, carriage returns, tabs with spaces
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)

    # Collapse multiple consecutive spaces
    cleaned = re.sub(r" {2,}", " ", cleaned)

    # Clean up orphaned punctuation spacing (e.g., "word , word" -> "word, word")
    cleaned = re.sub(r"\s+([,.\?!;:])", r"\1", cleaned)

    return cleaned.strip()
