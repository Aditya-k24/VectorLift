"""
VectorLift — Text Preprocessing Utilities
==========================================
Pure-function helpers for cleaning, normalising, truncating and chunking text.
These utilities are intentionally framework-agnostic — they have no imports
from the rest of the VectorLift codebase and can be used in any pipeline stage.

All functions accept and return plain Python strings; they never mutate the
input string in place.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterator


# Pre-compiled regexes (compiled once at module import for performance)
_RE_MULTI_WHITESPACE = re.compile(r"\s+")
_RE_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]")
_RE_UNICODE_QUOTES = re.compile(r"[‘’]")          # ' '
_RE_UNICODE_DOUBLE_QUOTES = re.compile(r'[“”]')   # " "
_RE_DASHES = re.compile(r"[–—]")                  # en-dash, em-dash
_RE_ELLIPSIS = re.compile(r"…")                        # …
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_URL = re.compile(
    r"https?://[^\s]+"
    r"|www\.[^\s]+"
    r"|ftp://[^\s]+",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


def clean_text(
    text: str,
    *,
    remove_html: bool = True,
    remove_urls: bool = False,
    normalise_unicode: bool = True,
    normalise_whitespace: bool = True,
    strip_control_chars: bool = True,
    lowercase: bool = False,
) -> str:
    """
    Apply a configurable chain of cleaning operations to ``text``.

    Parameters
    ----------
    text:
        Input string.
    remove_html:
        Strip HTML / XML tags.
    remove_urls:
        Replace URLs with a single space.
    normalise_unicode:
        Apply NFC normalisation and replace fancy quotes / dashes with their
        ASCII equivalents.
    normalise_whitespace:
        Collapse consecutive whitespace characters (including ``\\t``, ``\\n``)
        into a single space and strip leading / trailing whitespace.
    strip_control_chars:
        Remove ASCII control characters (non-printable bytes 0x00–0x1F except
        tab ``\\t``, newline ``\\n`` and carriage-return ``\\r``).
    lowercase:
        Convert the result to lowercase.

    Returns
    -------
    str
        Cleaned text string.
    """
    if not text:
        return ""

    if remove_html:
        text = _RE_HTML_TAG.sub(" ", text)

    if remove_urls:
        text = _RE_URL.sub(" ", text)

    if normalise_unicode:
        # NFC normalisation: compose combining characters
        text = unicodedata.normalize("NFC", text)
        # Typographic → ASCII equivalents
        text = _RE_UNICODE_QUOTES.sub("'", text)
        text = _RE_UNICODE_DOUBLE_QUOTES.sub('"', text)
        text = _RE_DASHES.sub("-", text)
        text = _RE_ELLIPSIS.sub("...", text)

    if strip_control_chars:
        text = _RE_CONTROL_CHARS.sub("", text)

    if normalise_whitespace:
        text = _RE_MULTI_WHITESPACE.sub(" ", text).strip()

    if lowercase:
        text = text.lower()

    return text


# ---------------------------------------------------------------------------
# truncate_text
# ---------------------------------------------------------------------------


def truncate_text(
    text: str,
    max_length: int,
    *,
    truncation_marker: str = "…",
    unit: str = "chars",
) -> str:
    """
    Truncate ``text`` to at most ``max_length`` units.

    Parameters
    ----------
    text:
        Input string.
    max_length:
        Maximum allowed length in the specified ``unit``.
    truncation_marker:
        String appended when truncation occurs (counts toward ``max_length``).
        Set to ``""`` to truncate without any marker.
    unit:
        ``"chars"``  — truncate by Unicode character count (default).
        ``"words"``  — truncate by whitespace-separated word count.
        ``"tokens"`` — approximate word-piece count (splits on whitespace and
                       punctuation); useful for rough token-budget estimates
                       before sending to a tokeniser.

    Returns
    -------
    str
        Original string (unchanged) or a truncated version with the marker.

    Raises
    ------
    ValueError
        If ``max_length < len(truncation_marker)`` (impossible to fit anything
        useful in the budget).
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be > 0, got {max_length}")

    marker_len = len(truncation_marker)
    if max_length < marker_len:
        raise ValueError(
            f"max_length ({max_length}) is too small to fit the truncation_marker "
            f"(length {marker_len})"
        )

    if unit == "chars":
        if len(text) <= max_length:
            return text
        return text[: max_length - marker_len] + truncation_marker

    elif unit == "words":
        words = text.split()
        if len(words) <= max_length:
            return text
        truncated = " ".join(words[: max_length - (1 if truncation_marker else 0)])
        return truncated + (" " + truncation_marker if truncation_marker else "")

    elif unit == "tokens":
        # Approximate tokenisation: split on whitespace and punctuation
        tokens = re.findall(r"\w+|[^\w\s]", text)
        if len(tokens) <= max_length:
            return text
        budget = max_length - (1 if truncation_marker else 0)
        truncated_tokens = tokens[:budget]
        result = _rejoin_tokens(truncated_tokens)
        return result + (" " + truncation_marker if truncation_marker else "")

    else:
        raise ValueError(f"Unknown unit: {unit!r}. Choose from 'chars', 'words', 'tokens'.")


def _rejoin_tokens(tokens: list[str]) -> str:
    """Re-join approximate tokens with sensible spacing."""
    result: list[str] = []
    for i, tok in enumerate(tokens):
        if i == 0:
            result.append(tok)
        elif re.match(r"[^\w]", tok):
            # Punctuation — no leading space
            result.append(tok)
        else:
            result.append(" " + tok)
    return "".join(result)


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int = 0,
    *,
    unit: str = "words",
    min_chunk_size: int = 1,
) -> list[str]:
    """
    Split ``text`` into overlapping chunks of approximately ``chunk_size`` units.

    Parameters
    ----------
    text:
        Input string to split.
    chunk_size:
        Target chunk size in the specified ``unit``.
    overlap:
        Number of units shared between consecutive chunks.  Must be strictly
        less than ``chunk_size``.
    unit:
        ``"words"``  — split on whitespace (default, fast).
        ``"chars"``  — split by character count.
        ``"sentences"`` — split by sentence boundaries.
    min_chunk_size:
        Discard trailing chunks smaller than this many units.

    Returns
    -------
    list[str]
        List of text chunks.  Empty list if ``text`` is blank.

    Raises
    ------
    ValueError
        If ``overlap >= chunk_size``.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be < chunk_size ({chunk_size})"
        )
    if not text or not text.strip():
        return []

    if unit == "words":
        return _chunk_by_tokens(text.split(), chunk_size, overlap, min_chunk_size, " ")

    elif unit == "chars":
        chars = list(text)
        raw_chunks = _chunk_sequence(chars, chunk_size, overlap, min_chunk_size)
        return ["".join(chunk) for chunk in raw_chunks]

    elif unit == "sentences":
        sentences = _split_sentences(text)
        return _chunk_by_tokens(sentences, chunk_size, overlap, min_chunk_size, " ")

    else:
        raise ValueError(f"Unknown unit: {unit!r}. Choose from 'words', 'chars', 'sentences'.")


def _chunk_by_tokens(
    tokens: list[str],
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
    sep: str,
) -> list[str]:
    raw_chunks = _chunk_sequence(tokens, chunk_size, overlap, min_chunk_size)
    return [sep.join(chunk) for chunk in raw_chunks]


def _chunk_sequence(
    seq: list[str],
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
) -> list[list[str]]:
    step = chunk_size - overlap
    chunks: list[list[str]] = []
    start = 0
    while start < len(seq):
        end = start + chunk_size
        chunk = seq[start:end]
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
        start += step
    return chunks


_RE_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")


def _split_sentences(text: str) -> list[str]:
    """Heuristic sentence splitter (no NLTK dependency)."""
    return [s.strip() for s in _RE_SENTENCE_BOUNDARY.split(text) if s.strip()]


# ---------------------------------------------------------------------------
# normalize_score
# ---------------------------------------------------------------------------


def normalize_score(
    score: float,
    min_val: float,
    max_val: float,
    *,
    clip: bool = True,
    epsilon: float = 1e-10,
) -> float:
    """
    Min-max normalise a score into the range ``[0.0, 1.0]``.

    Parameters
    ----------
    score:
        The raw score to normalise.
    min_val:
        The minimum value of the score range.
    max_val:
        The maximum value of the score range.
    clip:
        When ``True`` (default) clamp the result to ``[0.0, 1.0]`` even if
        ``score`` falls outside ``[min_val, max_val]``.
    epsilon:
        Small value added to the denominator to avoid division by zero when
        ``max_val == min_val``.

    Returns
    -------
    float
        Normalised score in ``[0.0, 1.0]``.
    """
    denominator = max_val - min_val + epsilon
    normalised = (score - min_val) / denominator

    if clip:
        return max(0.0, min(1.0, normalised))
    return normalised


def batch_normalize_scores(
    scores: list[float],
    *,
    clip: bool = True,
    epsilon: float = 1e-10,
) -> list[float]:
    """
    Normalise a list of scores using the observed min / max of the list.

    Parameters
    ----------
    scores:
        Raw scores to normalise.
    clip, epsilon:
        Forwarded to :func:`normalize_score`.

    Returns
    -------
    list[float]
        Normalised scores in the same order as the input.
    """
    if not scores:
        return []
    min_val = min(scores)
    max_val = max(scores)
    return [normalize_score(s, min_val, max_val, clip=clip, epsilon=epsilon) for s in scores]


# ---------------------------------------------------------------------------
# Additional convenience helpers
# ---------------------------------------------------------------------------


def word_count(text: str) -> int:
    """Return the number of whitespace-separated words in ``text``."""
    return len(text.split())


def char_count(text: str, *, include_spaces: bool = True) -> int:
    """Return the character count, optionally excluding spaces."""
    if include_spaces:
        return len(text)
    return len(text.replace(" ", ""))


def iter_chunks(
    text: str,
    chunk_size: int,
    overlap: int = 0,
    *,
    unit: str = "words",
) -> Iterator[str]:
    """Generator version of :func:`chunk_text`."""
    yield from chunk_text(text, chunk_size, overlap, unit=unit)
