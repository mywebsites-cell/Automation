from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class StoryPart:
    index: int
    part_title: str
    text: str
    cliffhanger: str


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def _chunk_words(words: list[str], min_words: int, max_words: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    i = 0
    n = len(words)
    while i < n:
        remaining = n - i
        if remaining <= max_words:
            chunk_size = remaining
        else:
            chunk_size = min(max_words, max(min_words, remaining // max(1, (remaining // max_words))))
        chunks.append(words[i : i + chunk_size])
        i += chunk_size
    return chunks


def _ensure_target_word_range(chunks: list[list[str]], min_words: int, max_words: int) -> list[list[str]]:
    if len(chunks) <= 1:
        return chunks
    for idx in range(len(chunks) - 1):
        while len(chunks[idx]) < min_words and len(chunks[idx + 1]) > min_words:
            chunks[idx].append(chunks[idx + 1].pop(0))
        while len(chunks[idx]) > max_words and len(chunks[idx + 1]) < max_words:
            chunks[idx + 1].insert(0, chunks[idx].pop())
    return chunks


def split_story_into_parts(
    story_text: str,
    min_words: int = 130,
    max_words: int = 160,
    max_parts: int = 10,
) -> list[StoryPart]:
    normalized = _normalize_text(story_text)
    words = normalized.split(" ")
    if not words:
        return []

    chunks = _chunk_words(words, min_words=min_words, max_words=max_words)
    chunks = _ensure_target_word_range(chunks, min_words=min_words, max_words=max_words)

    if len(chunks) > max_parts:
        merged: list[list[str]] = []
        stride = len(chunks) / max_parts
        start = 0.0
        for _ in range(max_parts):
            end = start + stride
            slice_start = int(round(start))
            slice_end = max(slice_start + 1, int(round(end)))
            merged_words: list[str] = []
            for chunk in chunks[slice_start:slice_end]:
                merged_words.extend(chunk)
            if merged_words:
                merged.append(merged_words)
            start = end
        chunks = merged

    parts: list[StoryPart] = []
    for i, chunk in enumerate(chunks, start=1):
        body = " ".join(chunk).strip()
        body = body.rstrip(". ")
        cliffhanger = "...but what happened next changed everything."
        if i == len(chunks):
            cliffhanger = "...and that is how the full story finally unfolded."
        full_text = f"{body}. {cliffhanger}"
        parts.append(
            StoryPart(
                index=i,
                part_title=f"Part {i}: The Turning Point",
                text=full_text,
                cliffhanger=cliffhanger,
            )
        )
    return parts
