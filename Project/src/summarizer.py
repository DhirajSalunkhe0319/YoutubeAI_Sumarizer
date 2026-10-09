"""
AI Summarization module using open-source google/flan-t5-base from Hugging Face.
Supports CUDA GPU acceleration and CPU fallback.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "google/flan-t5-base"

SUMMARIZATION_PROMPT_TEMPLATE = """You are an expert research analyst. Provide a comprehensive, detailed executive summary of the following YouTube transcript chunk.

Analyze and synthesize the following:
- Core Concepts & Frameworks: Detail the foundational concepts, definitions, and main arguments thoroughly.
- Mechanisms & Evidence: Explain how the process or system works, including critical facts, demonstrations, and examples.
- Actionable Takeaways & Significance: Summarize the conclusions, implications, and key lessons.

Provide rich, multi-sentence explanations that fully articulate the concepts rather than brief one-liners.

Transcript:
{chunk}"""


def get_device() -> torch.device:
    """
    Detects hardware availability and returns torch.device('cuda') or torch.device('cpu').

    Returns:
        torch.device: Active computing device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_name() -> str:
    """
    Returns a human-readable name of the active compute device.

    Returns:
        str: Device description (e.g. 'CUDA (NVIDIA GeForce RTX ...)' or 'CPU').
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return f"CUDA ({gpu_name})"
    return "CPU"


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[torch.device] = None,
) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM, torch.device]:
    """
    Loads and caches the Hugging Face tokenizer and Seq2Seq model.

    Args:
        model_name: Hugging Face model identifier (default: 'google/flan-t5-base').
        device: Target torch device. If None, automatically detected.

    Returns:
        Tuple[AutoTokenizer, AutoModelForSeq2SeqLM, torch.device]:
            Initialized tokenizer, model in evaluation mode, and target device.
    """
    if device is None:
        device = get_device()

    logger.info("Loading tokenizer for model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Loading model weights for model: %s on device: %s", model_name, device)
    # Using float16 if on CUDA for faster inference and lower VRAM usage, float32 on CPU
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    return tokenizer, model, device


def summarize_chunk(
    chunk: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    max_new_tokens: int = 256,
    min_new_tokens: int = 45,
    num_beams: int = 4,
) -> str:
    """
    Summarizes a single transcript chunk using FLAN-T5 beam search generation.

    Args:
        chunk: Text chunk (approx. 300 words).
        tokenizer: Pre-loaded AutoTokenizer.
        model: Pre-loaded AutoModelForSeq2SeqLM.
        device: torch.device ('cuda' or 'cpu').
        max_new_tokens: Maximum tokens in generated summary (default: 256).
        min_new_tokens: Minimum tokens in generated summary (default: 45).
        num_beams: Number of beams for beam search (default: 4).

    Returns:
        str: Generated summary text.
    """
    if not chunk or not chunk.strip():
        return ""

    prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(chunk=chunk.strip())

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    # Move tensors to destination device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            length_penalty=1.2,
            no_repeat_ngram_size=3,
        )

    summary_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary_text.strip()


def summarize_chunks(
    chunks: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    max_new_tokens: int = 256,
    min_new_tokens: int = 45,
    num_beams: int = 4,
) -> List[str]:
    """
    Sequentially summarizes a list of transcript chunks, providing progress callbacks.

    Args:
        chunks: List of transcript chunks.
        tokenizer: Pre-loaded tokenizer.
        model: Pre-loaded FLAN-T5 model.
        device: Compute device.
        progress_callback: Optional callback func(chunk_index, total_chunks, current_summary).
        max_new_tokens: Max generated tokens per chunk.
        min_new_tokens: Min generated tokens per chunk.
        num_beams: Beam search beam count.

    Returns:
        List[str]: List of summaries, one for each chunk.
    """
    if not chunks:
        return []

    total_chunks = len(chunks)
    chunk_summaries: List[str] = []

    for idx, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(idx + 1, total_chunks, f"Summarizing chunk {idx + 1} of {total_chunks}...")

        summary = summarize_chunk(
            chunk=chunk,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
        )

        chunk_summaries.append(summary)

    return chunk_summaries
