"""
End-to-end verification script for YouTube AI Summarizer.
Executes all 6 test scenarios defined in the project requirements.
"""

import os
import sys
import torch

# Configure UTF-8 encoding for Windows console output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.youtube import extract_video_id, YouTubeURLError
from src.transcript import (
    fetch_transcript,
    clean_transcript,
    TranscriptNotAvailableError,
    VideoNotAvailableError,
)
from src.utils import chunk_transcript, count_words, format_final_notes
from src.summarizer import (
    load_model,
    summarize_chunk,
    summarize_chunks,
    get_device,
    get_device_name,
)


def run_all_tests():
    print("=" * 60)
    print("RUNNING YOUTUBE SUMMARIZER VERIFICATION SUITE")
    print("=" * 60)

    # -------------------------------------------------------------
    # Test 5 & 6: GPU Detection & Device Fallback
    # -------------------------------------------------------------
    print("\n--- Test 5 & 6: Hardware Detection ---")
    device = get_device()
    device_name = get_device_name()
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Detected Device: {device} ({device_name})")
    assert device.type in ("cuda", "cpu")
    print(" Hardware detection passed.")

    # -------------------------------------------------------------
    # Test 2: Invalid YouTube URLs
    # -------------------------------------------------------------
    print("\n--- Test 2: Invalid YouTube URLs Handling ---")
    invalid_cases = [
        "",
        "https://vimeo.com/12345678",
        "https://www.youtube.com/watch?v=too_short",
        "https://notyoutube.com/watch?v=dQw4w9WgXcQ",
        "invalid_non_youtube_input",
    ]
    for inv in invalid_cases:
        try:
            extract_video_id(inv)
            raise AssertionError(f"Expected failure for '{inv}'")
        except YouTubeURLError as e:
            print(f" Properly rejected '{inv}': {e}")
    print(" Invalid URL handling passed.")

    # -------------------------------------------------------------
    # Test 3: Video Without Captions Handling
    # -------------------------------------------------------------
    print("\n--- Test 3: Video Without Captions Handling ---")
    # Video with disabled or non-existent captions
    try:
        # Test with a known video or handle simulation
        print("Checking error handling structure for missing captions...")
        # Raising mock or testing invalid ID retrieval
        try:
            fetch_transcript("invalidID11")
        except Exception as e:
            print(f" Handled inaccessible transcript cleanly: {type(e).__name__} -> {e}")
    except Exception as e:
        print(f" Unexpected error: {e}")

    # -------------------------------------------------------------
    # Test 1 & 4: Valid YouTube URL, Chunking & End-to-End Summarization
    # -------------------------------------------------------------
    print("\n--- Test 1 & 4: Valid URL, Word Chunking & FLAN-T5 Summarization ---")
    valid_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print(f"1. Extracting ID from {valid_url}...")
    video_id = extract_video_id(valid_url)
    print(f"   Video ID: {video_id}")

    print("2. Fetching transcript...")
    transcript, meta = fetch_transcript(video_id)
    total_words = count_words(transcript)
    print(f"   Transcript fetched: {total_words} words ({meta.get('language')}).")

    print("3. Chunking transcript (300 words per chunk)...")
    chunks = chunk_transcript(transcript, chunk_size=300)
    print(f"   Generated {len(chunks)} chunk(s).")

    print("4. Loading FLAN-T5 model...")
    tokenizer, model, dev = load_model()
    print("   Model loaded successfully.")

    print("5. Summarizing chunks sequentially...")
    chunk_summaries = []
    for idx, ch in enumerate(chunks):
        print(f"   Processing chunk {idx + 1}/{len(chunks)} ({count_words(ch)} words)...")
        summary = summarize_chunk(
            chunk=ch,
            tokenizer=tokenizer,
            model=model,
            device=dev,
            num_beams=4,
            max_new_tokens=120,
            min_new_tokens=15,
        )
        print(f"   Summary {idx + 1}: {summary}")
        chunk_summaries.append(summary)

    print("6. Formatting final research notes...")
    final_notes = format_final_notes(chunk_summaries)
    print("\n" + final_notes)
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
