"""
YouTube Video Summarizer
An open-source web application for extracting transcripts and generating concise
research notes using local Hugging Face models.
"""

__version__ = "1.0.0"

from src.youtube import YouTubeURLError, extract_video_id, fetch_video_metadata
from src.transcript import fetch_transcript, clean_transcript
from src.utils import chunk_transcript, count_words, format_final_notes
from src.summarizer import load_model, summarize_chunk
