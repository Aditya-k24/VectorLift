"""
Unit tests for core/utils/text.py.
"""
from __future__ import annotations

import pytest

from core.utils.text import (
    chunk_text,
    clean_text,
    normalize_score,
    truncate_text,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_removes_html_tags(self):
        result = clean_text("<p>Hello <b>World</b></p>", remove_html=True)
        assert "<" not in result
        assert ">" not in result
        assert "Hello" in result
        assert "World" in result

    def test_strips_nested_html(self):
        result = clean_text("<div><span class='x'>text</span></div>")
        assert result.strip() == "text"

    def test_preserves_text_when_remove_html_false(self):
        html = "<b>bold</b>"
        result = clean_text(html, remove_html=False)
        assert "<b>" in result

    def test_normalises_whitespace_collapses_spaces(self):
        result = clean_text("hello   world", normalise_whitespace=True)
        assert result == "hello world"

    def test_normalises_whitespace_collapses_newlines(self):
        result = clean_text("hello\n\nworld\t !", normalise_whitespace=True)
        assert "  " not in result
        assert "\n" not in result

    def test_strips_leading_trailing_whitespace(self):
        result = clean_text("   hello   ", normalise_whitespace=True)
        assert result == "hello"

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_lowercase_option(self):
        result = clean_text("Hello WORLD", lowercase=True)
        assert result == "hello world"

    def test_remove_urls(self):
        text = "Visit https://example.com for more info"
        result = clean_text(text, remove_urls=True)
        assert "https://" not in result
        assert "example.com" not in result
        assert "Visit" in result

    def test_preserves_url_when_remove_urls_false(self):
        url = "https://example.com"
        result = clean_text(url, remove_urls=False)
        assert "example" in result

    def test_unicode_quote_normalisation(self):
        result = clean_text("‘curly’", normalise_unicode=True)
        assert "‘" not in result
        assert "curly" in result

    def test_html_plus_whitespace(self):
        result = clean_text("<p>  spaced  </p>  text  ")
        assert result == "spaced text"


# ---------------------------------------------------------------------------
# truncate_text
# ---------------------------------------------------------------------------


class TestTruncateText:
    def test_truncate_chars_no_truncation_needed(self):
        text = "hello"
        result = truncate_text(text, max_length=10, unit="chars")
        assert result == text

    def test_truncate_chars_truncates(self):
        text = "hello world"
        result = truncate_text(text, max_length=7, unit="chars", truncation_marker="…")
        # 7 chars total: 6 of text + 1 marker
        assert len(result) == 7
        assert result.endswith("…")

    def test_truncate_chars_no_marker(self):
        text = "hello world"
        result = truncate_text(text, max_length=5, unit="chars", truncation_marker="")
        assert result == "hello"
        assert len(result) == 5

    def test_truncate_words_no_truncation_needed(self):
        text = "one two three"
        result = truncate_text(text, max_length=5, unit="words")
        assert result == text

    def test_truncate_words_truncates(self):
        text = "one two three four five six"
        result = truncate_text(text, max_length=3, unit="words")
        words = result.split()
        # Should contain at most 3 content words (plus potential marker word)
        assert "one" in result
        assert "two" in result
        assert "six" not in result

    def test_truncate_words_marker_added(self):
        text = "one two three four five"
        result = truncate_text(text, max_length=2, unit="words", truncation_marker="…")
        assert "…" in result

    def test_invalid_max_length_raises(self):
        with pytest.raises(ValueError):
            truncate_text("hello", max_length=0, unit="chars")

    def test_negative_max_length_raises(self):
        with pytest.raises(ValueError):
            truncate_text("hello", max_length=-1)

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            truncate_text("hello", max_length=5, unit="paragraphs")

    def test_exact_length_not_truncated(self):
        text = "abcde"
        result = truncate_text(text, max_length=5, unit="chars", truncation_marker="…")
        assert result == text


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_chunk_words_no_overlap(self):
        text = "one two three four five six"
        chunks = chunk_text(text, chunk_size=2, overlap=0, unit="words")
        assert len(chunks) == 3
        assert chunks[0] == "one two"
        assert chunks[1] == "three four"
        assert chunks[2] == "five six"

    def test_chunk_words_with_overlap(self):
        text = "a b c d e"
        chunks = chunk_text(text, chunk_size=3, overlap=1, unit="words")
        # chunks: [a b c], [c d e]
        assert len(chunks) == 2
        assert "c" in chunks[0]
        assert "c" in chunks[1]

    def test_chunk_chars_no_overlap(self):
        text = "abcdef"
        chunks = chunk_text(text, chunk_size=2, overlap=0, unit="chars")
        assert len(chunks) == 3
        assert chunks[0] == "ab"
        assert chunks[1] == "cd"
        assert chunks[2] == "ef"

    def test_chunk_chars_with_overlap(self):
        text = "abcde"
        chunks = chunk_text(text, chunk_size=3, overlap=1, unit="chars")
        # [abc], [cde]
        assert len(chunks) == 2
        assert chunks[0] == "abc"
        assert chunks[1] == "cde"

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("", chunk_size=5, unit="words") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   ", chunk_size=3, unit="words") == []

    def test_overlap_ge_chunk_size_raises(self):
        with pytest.raises(ValueError):
            chunk_text("hello world", chunk_size=2, overlap=2)

    def test_text_smaller_than_chunk_returns_single_chunk(self):
        text = "hello world"
        chunks = chunk_text(text, chunk_size=100, unit="words")
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            chunk_text("hello", chunk_size=1, unit="paragraphs")

    def test_min_chunk_size_filters_small_trailing_chunk(self):
        text = "a b c d e"
        # chunk_size=3, overlap=0, step=3: [a b c], [d e]
        # min_chunk_size=3 should discard [d e]
        chunks = chunk_text(text, chunk_size=3, overlap=0, unit="words", min_chunk_size=3)
        assert len(chunks) == 1
        assert chunks[0] == "a b c"


# ---------------------------------------------------------------------------
# normalize_score
# ---------------------------------------------------------------------------


class TestNormalizeScore:
    def test_midpoint_is_0_5(self):
        result = normalize_score(5.0, min_val=0.0, max_val=10.0)
        assert abs(result - 0.5) < 1e-6

    def test_min_val_maps_to_near_0(self):
        result = normalize_score(0.0, min_val=0.0, max_val=10.0)
        assert result < 0.01

    def test_max_val_maps_to_near_1(self):
        result = normalize_score(10.0, min_val=0.0, max_val=10.0)
        assert result > 0.99

    def test_clip_prevents_above_1(self):
        result = normalize_score(20.0, min_val=0.0, max_val=10.0, clip=True)
        assert result <= 1.0

    def test_clip_prevents_below_0(self):
        result = normalize_score(-5.0, min_val=0.0, max_val=10.0, clip=True)
        assert result >= 0.0

    def test_no_clip_allows_outside_range(self):
        result = normalize_score(20.0, min_val=0.0, max_val=10.0, clip=False)
        assert result > 1.0

    def test_degenerate_range_returns_nonzero(self):
        # min_val == max_val — epsilon prevents division by zero
        result = normalize_score(5.0, min_val=5.0, max_val=5.0, clip=True)
        assert 0.0 <= result <= 1.0

    def test_negative_range(self):
        result = normalize_score(-5.0, min_val=-10.0, max_val=0.0)
        assert abs(result - 0.5) < 0.01
