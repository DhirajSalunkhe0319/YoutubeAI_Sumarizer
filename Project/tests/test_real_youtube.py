import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.youtube import extract_video_id, YouTubeURLError
from src.transcript import fetch_transcript, TranscriptNotAvailableError
from src.utils import chunk_transcript, count_words, format_final_notes


def test_real_video():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print(f"Testing URL: {url}")
    video_id = extract_video_id(url)
    print(f"Extracted Video ID: {video_id}")
    assert video_id == "dQw4w9WgXcQ"

    text, meta = fetch_transcript(video_id)
    print(f"Fetched transcript: {len(text)} characters, {count_words(text)} words.")
    print(f"Metadata: {meta}")
    assert len(text) > 0

    chunks = chunk_transcript(text, chunk_size=300)
    print(f"Chunks generated: {len(chunks)}")
    assert len(chunks) >= 1
    print("Integration test passed successfully!")


if __name__ == "__main__":
    test_real_video()
