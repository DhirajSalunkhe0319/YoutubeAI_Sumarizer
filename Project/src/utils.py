"""
Utility functions for text chunking, word counting, and summary formatting.
"""

from __future__ import annotations

import re
from typing import List


def count_words(text: str) -> int:
    """
    Counts the number of words in a given string.

    Args:
        text: Input text string.

    Returns:
        int: Total word count.
    """
    if not text or not text.strip():
        return 0
    return len(text.strip().split())


def chunk_transcript(text: str, chunk_size: int = 300) -> List[str]:
    """
    Splits transcript text into word-based chunks of approximately `chunk_size` words.
    Word-based chunking is crucial because auto-generated YouTube captions frequently
    lack reliable punctuation or sentence boundaries.

    Args:
        text: Cleaned transcript text.
        chunk_size: Maximum target number of words per chunk (default: 300).

    Returns:
        List[str]: List of transcript chunks.
    """
    if not text or not text.strip():
        return []

    words = text.strip().split()
    if not words:
        return []

    if chunk_size <= 0:
        chunk_size = 300

    chunks: List[str] = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))

    return chunks


def format_bullet_points(raw_summary: str) -> List[str]:
    """
    Extracts and standardizes bullet points or key sentences from a raw summary string.

    Args:
        raw_summary: Raw generated summary string from the model.

    Returns:
        List[str]: List of clean bullet-point sentences.
    """
    if not raw_summary or not raw_summary.strip():
        return []

    lines = raw_summary.strip().split("\n")
    bullet_items: List[str] = []

    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue

        # Strip existing bullet markers (-, *, •, numbers like 1.)
        cleaned_line = re.sub(r"^(\s*[-*•]|\d+[\.)])\s*", "", cleaned_line).strip()
        if not cleaned_line:
            continue

        # Split multiple sentences into clean bullet items
        sentences = re.split(r"(?<=[.!?])\s+", cleaned_line)
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 5 and s_clean not in bullet_items:
                bullet_items.append(s_clean)

    return bullet_items


def format_transcript_paragraphs(text: str, target_sentences: int = 3) -> str:
    """
    Formats a transcript into readable paragraph blocks while preserving the original
    wording, meaning, and chronology. If the transcript already contains paragraph breaks,
    they are preserved; otherwise sentences are grouped into natural paragraph chunks.
    """
    if not text or not text.strip():
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", normalized.strip()) if block.strip()]
    if not blocks:
        blocks = [normalized.strip()]

    paragraphs: List[str] = []
    for block in blocks:
        block = re.sub(r"\s+", " ", block).strip()
        if not block:
            continue

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", block) if s.strip()]
        if not sentences:
            paragraphs.append(block)
            continue

        current: List[str] = []
        for sentence in sentences:
            current.append(sentence)
            if len(current) >= target_sentences:
                paragraphs.append(" ".join(current).strip())
                current = []

        if current:
            paragraphs.append(" ".join(current).strip())

    return "\n\n".join(paragraphs)


def format_final_notes(chunk_summaries: List[str]) -> str:
    """
    Combines individual chunk summaries into a cohesive, beautifully structured
    Markdown notes document.

    Args:
        chunk_summaries: List of generated summary strings for each chunk.

    Returns:
        str: Clean Markdown formatted summary notes.
    """
    if not chunk_summaries:
        return "No summary could be generated."

    all_points: List[str] = []
    for summary in chunk_summaries:
        points = format_bullet_points(summary)
        for pt in points:
            if pt not in all_points:
                all_points.append(pt)

    if not all_points:
        return "## 📝 Video Summary\n\nNo significant takeaways could be extracted."

    markdown_output = ["## 📝 Video Summary\n"]
    for point in all_points:
        # Ensure capitalization and ending period
        formatted_pt = point[0].upper() + point[1:] if len(point) > 1 else point.upper()
        if not formatted_pt.endswith((".", "!", "?")):
            formatted_pt += "."
        markdown_output.append(f"- {formatted_pt}")

    return "\n".join(markdown_output)


def format_timestamp(seconds: float) -> str:
    """
    Formats a duration in seconds into MM:SS or HH:MM:SS format.

    Args:
        seconds: Duration in seconds.

    Returns:
        str: Formatted timestamp string (e.g. '04:15' or '01:12:30').
    """
    total_seconds = int(max(0, seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def chunk_snippets_with_timestamps(snippets: List[dict], chunk_size: int = 300) -> List[dict]:
    """
    Groups raw timestamped snippets into word-based chunks while tracking the chunk's start time.

    Args:
        snippets: List of dicts with 'text', 'start', 'duration'.
        chunk_size: Target words per chunk (default: 300).

    Returns:
        List[dict]: List of chunk dicts:
            [{'text': str, 'start': float, 'start_formatted': str, 'word_count': int}]
    """
    if not snippets:
        return []

    chunks = []
    current_words = []
    current_start = None

    for snippet in snippets:
        text = snippet.get("text", "").strip()
        if not text:
            continue

        words = text.split()
        if not words:
            continue

        if current_start is None:
            current_start = snippet.get("start", 0.0)

        current_words.extend(words)

        if len(current_words) >= chunk_size:
            chunk_text = " ".join(current_words)
            chunks.append({
                "text": chunk_text,
                "start": current_start,
                "start_formatted": format_timestamp(current_start),
                "word_count": len(current_words),
            })
            current_words = []
            current_start = None

    if current_words:
        chunk_text = " ".join(current_words)
        chunks.append({
            "text": chunk_text,
            "start": current_start or 0.0,
            "start_formatted": format_timestamp(current_start or 0.0),
            "word_count": len(current_words),
        })

    return chunks


def parse_executive_notes(chunk_summaries: List[str]) -> List[dict]:
    """
    Parses and structures chunk summaries into detailed executive note cards matching the UI theme.
    Produces comprehensive, multi-sentence explanations detailing core concepts and mechanisms.

    Args:
        chunk_summaries: Generated summary strings.

    Returns:
        List[dict]: List of {'number': int, 'title': str, 'content': str}.
    """
    raw_points: List[str] = []
    for summary in chunk_summaries:
        points = format_bullet_points(summary)
        for pt in points:
            if pt and pt not in raw_points:
                raw_points.append(pt)

    # Combine short adjacent points into fuller notes that read like real explanations.
    grouped_items: List[str] = []
    i = 0
    while i < len(raw_points):
        combined = [raw_points[i].strip()]
        next_index = i + 1
        while next_index < len(raw_points):
            next_point = raw_points[next_index].strip()
            candidate = " ".join(combined + [next_point])
            if len(candidate.split()) <= 45 and len(next_point.split()) <= 22:
                combined.append(next_point)
                next_index += 1
                if len(combined) >= 3:
                    break
            else:
                break
        grouped_items.append(" ".join(combined))
        i = next_index

    notes = []
    for idx, item in enumerate(grouped_items, start=1):
        clean_item = item.strip()

        # Prefer a title-like phrase for the first sentence, while keeping the rest of the note rich.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_item) if s.strip()]
        if not sentences:
            sentences = [clean_item]

        if ":" in clean_item and not clean_item.startswith("http"):
            parts = clean_item.split(":", 1)
            title = parts[0].strip().rstrip(".")
            content = parts[1].strip()
        elif " - " in clean_item:
            parts = clean_item.split(" - ", 1)
            title = parts[0].strip().rstrip(".")
            content = parts[1].strip()
        elif len(sentences) > 1:
            first_sentence = sentences[0].strip().rstrip(".?!")
            words = first_sentence.split()
            if len(words) <= 4:
                title = "Key Insight"
            else:
                title = " ".join(words[:4]).strip().rstrip(".")
            content = " ".join(sentences)
        else:
            words = clean_item.split()
            if len(words) <= 4:
                title = "Key Insight"
                content = clean_item
            elif "," in clean_item and clean_item.find(",") < 32:
                c_idx = clean_item.find(",")
                title = clean_item[:c_idx].strip().rstrip(".")
                content = clean_item[c_idx + 1:].strip()
            else:
                title = " ".join(words[:3]).strip().rstrip(".")
                content = " ".join(words[3:]).strip()

        # Keep content as a fuller explanation when possible.
        if content:
            content = re.sub(r"\s+", " ", content).strip()
            content = content[0].upper() + content[1:]
            if not content.endswith((".", "!", "?")):
                content += "."
        if title:
            title = re.sub(r"\s+", " ", title).strip()
            title = title[0].upper() + title[1:]

        notes.append({
            "number": idx,
            "title": title,
            "content": content,
        })

    return notes


def estimate_reading_time(word_count: int, words_per_minute: int = 200) -> str:
    """
    Estimates reading time for a given word count.

    Args:
        word_count: Total words.
        words_per_minute: Standard reading speed (default 200 wpm).

    Returns:
        str: Human-readable reading time estimate (e.g. '1 min read').
    """
    if word_count <= 0:
        return "< 1 min read"
    minutes = max(1, round(word_count / words_per_minute))
    return f"~{minutes} min read"

