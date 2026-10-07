"""Deterministic document chunking shared by ingestion and runtime publishing.

The chunker works on plain text so it can be used for generated JSONL records,
Markdown files, and administrator-approved feedback without pulling a parser into
the request path. It keeps paragraphs together when possible, splits oversized
paragraphs on sentence boundaries, and applies a small character overlap so a fact
near a boundary remains retrievable from either side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_HEADING = re.compile(r"^\s{0,3}(?:#{1,6}\s+|【[^】]+】|\[[^\]]+\])")
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;])")


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str
    start: int
    end: int


def _normalise(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _paragraph_units(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # A heading is metadata for the following paragraph, not a standalone
        # retrieval result. Keep it attached unless the following text is absent.
        lines = paragraph.splitlines()
        if len(lines) > 1 and _HEADING.match(lines[0]):
            paragraph = f"{lines[0].strip()}\n{' '.join(line.strip() for line in lines[1:] if line.strip())}"
        else:
            paragraph = " ".join(line.strip() for line in lines if line.strip())
        units.append(paragraph)
    return units


def _split_oversized(unit: str, max_size: int) -> list[str]:
    if len(unit) <= max_size:
        return [unit]
    sentences = [part.strip() for part in _SENTENCE_END.split(unit) if part.strip()]
    if not sentences:
        sentences = [unit]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_size:
            if current:
                pieces.append(current.strip())
                current = ""
            pieces.extend(sentence[i : i + max_size] for i in range(0, len(sentence), max_size))
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_size:
            pieces.append(current.strip())
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current.strip())
    return pieces


def split_document(text: str, *, target_size: int = 800, overlap: int = 120, max_size: int = 1200) -> list[TextChunk]:
    """Split a document into overlapping, bounded chunks.

    Sizes are measured in Unicode characters, which is a practical approximation
    for mixed Chinese/English scenic-area content. ``overlap`` is applied only
    between chunks and never creates a chunk consisting solely of overlap text.
    """
    if target_size <= 0 or max_size < target_size or overlap < 0 or overlap >= target_size:
        raise ValueError("require 0 <= overlap < target_size <= max_size")
    normalised = _normalise(text)
    if not normalised:
        return []
    units: list[str] = []
    for unit in _paragraph_units(normalised):
        units.extend(_split_oversized(unit, max_size))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if current and len(candidate) > target_size:
            chunks.append(current.strip())
            tail = current[-overlap:].strip()
            current = f"{tail}\n\n{unit}".strip() if tail else unit
            # A paragraph plus overlap can exceed max_size; split it once more.
            if len(current) > max_size:
                chunks.extend(_split_oversized(current, max_size)[:-1])
                current = _split_oversized(current, max_size)[-1]
        else:
            current = candidate
    if current:
        chunks.append(current.strip())

    result: list[TextChunk] = []
    cursor = 0
    for index, content in enumerate(chunks):
        start = normalised.find(content[: min(40, len(content))], cursor)
        start = cursor if start < 0 else start
        end = min(len(normalised), start + len(content))
        result.append(TextChunk(index=index, content=content, start=start, end=end))
        cursor = max(cursor, end - overlap)
    return result

