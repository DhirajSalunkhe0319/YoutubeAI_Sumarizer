"""
Comprehensive unit and integration test suite for YouTube Summarizer.
Tests URL extraction, transcript retrieval & cleaning, word chunking, and AI summarization.
"""

import unittest
from src.youtube import extract_video_id, YouTubeURLError
from src.transcript import clean_transcript
from src.utils import chunk_transcript, count_words, format_final_notes, format_bullet_points, parse_executive_notes
from src.summarizer import load_model, summarize_chunk, get_device


class TestYouTubeURLExtraction(unittest.TestCase):
    """Test suite for YouTube URL extraction and validation."""

    def test_standard_watch_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")

    def test_shortened_youtu_be_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")

    def test_embed_url(self):
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")

    def test_shorts_url(self):
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")

    def test_url_with_query_parameters(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120s&feature=share&si=abc123"
        self.assertEqual(extract_video_id(url), "dQw4w9WgXcQ")

    def test_raw_id(self):
        self.assertEqual(extract_video_id("dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_invalid_url(self):
        with self.assertRaises(YouTubeURLError):
            extract_video_id("https://vimeo.com/12345678")

    def test_empty_url(self):
        with self.assertRaises(YouTubeURLError):
            extract_video_id("")

    def test_malformed_id(self):
        with self.assertRaises(YouTubeURLError):
            extract_video_id("https://www.youtube.com/watch?v=too_short")


class TestTranscriptCleaning(unittest.TestCase):
    """Test suite for transcript cleaning."""

    def test_cleaning_music_cues_and_whitespace(self):
        raw = "Hello world! [Music] Today we are discussing AI.   \n\n[Applause]   It is great."
        cleaned = clean_transcript(raw)
        self.assertNotIn("[Music]", cleaned)
        self.assertNotIn("[Applause]", cleaned)
        self.assertIn("Hello world!", cleaned)
        self.assertIn("Today we are discussing AI.", cleaned)
        self.assertNotIn("  ", cleaned)  # No duplicate spaces


class TestChunking(unittest.TestCase):
    """Test suite for word-based chunking."""

    def test_empty_chunking(self):
        self.assertEqual(chunk_transcript(""), [])

    def test_short_text(self):
        text = "This is a short transcript with only nine words."
        chunks = chunk_transcript(text, chunk_size=300)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_multichunk_text(self):
        # 650 words with chunk size 300 -> 3 chunks (300, 300, 50)
        words = [f"word{i}" for i in range(650)]
        text = " ".join(words)
        chunks = chunk_transcript(text, chunk_size=300)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0].split()), 300)
        self.assertEqual(len(chunks[1].split()), 300)
        self.assertEqual(len(chunks[2].split()), 50)


class TestFormatting(unittest.TestCase):
    """Test suite for notes formatting."""

    def test_bullet_formatting(self):
        raw_summary = "First major point. Second key discovery."
        notes = format_final_notes([raw_summary])
        self.assertIn("## 📝 Video Summary", notes)
        self.assertIn("- First major point.", notes)
        self.assertIn("- Second key discovery.", notes)

    def test_executive_notes_are_detailed(self):
        summaries = [
            "The team starts with a clear goal. They identify the audience and map the problem. They test a small prototype early.",
            "The workflow reduces wasted effort. It creates short feedback loops and keeps the project focused on learning."
        ]
        notes = parse_executive_notes(summaries)
        self.assertTrue(notes)
        self.assertTrue(all(len(note["content"].split()) >= 15 for note in notes))
        self.assertTrue(any("feedback" in note["content"].lower() for note in notes))
        self.assertTrue(all("Key Insight" not in note["title"] for note in notes))


class TestFLANT5Inference(unittest.TestCase):
    """Test suite for local model inference."""

    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        cls.tokenizer, cls.model, _ = load_model(device=cls.device)

    def test_summarization(self):
        sample = (
            "Machine learning algorithms build a model based on training data in order to make "
            "predictions or decisions without being explicitly programmed to do so. "
            "Supervised learning algorithms build a mathematical model of a set of data that contains "
            "both the inputs and the desired outputs. Unsupervised learning algorithms take a set of "
            "data that contains only inputs and find structure in the data."
        )
        summary = summarize_chunk(
            chunk=sample,
            tokenizer=self.tokenizer,
            model=self.model,
            device=self.device,
            max_new_tokens=80,
            min_new_tokens=15,
            num_beams=4,
        )
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary.strip()), 10)
        print("\n[Test Passed] Model generated summary:", summary)


if __name__ == "__main__":
    unittest.main()
