"""
Tests for methylbert.data.nanopore.chunker.chunk_read.
"""

import numpy as np
import pytest

from methylbert.data.nanopore.chunker import (
    chunk_read,
    WINDOW_LEN,
    MIN_TAIL,
    DEFAULT_PAD_TOKEN,
    DEFAULT_PAD_STATE,
)


def _arr(length, fill_token=7, fill_state=1):
    """Helper: parallel arrays with predictable distinct fills."""
    tokens = np.arange(length, dtype=np.int32) + fill_token
    states = np.full(length, fill_state, dtype=np.int8)
    return tokens, states


# ---- Length math --------------------------------------------------------

def test_empty_input_yields_nothing():
    tokens, states = _arr(0)
    assert list(chunk_read(tokens, states)) == []


def test_input_shorter_than_min_tail_yields_nothing():
    tokens, states = _arr(MIN_TAIL - 1)
    assert list(chunk_read(tokens, states)) == []


def test_exactly_one_window_yields_one_window():
    tokens, states = _arr(WINDOW_LEN)
    chunks = list(chunk_read(tokens, states))
    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0][0], tokens)
    np.testing.assert_array_equal(chunks[0][1], states)


def test_window_plus_short_tail_drops_tail():
    """One full window + a tail shorter than min_tail -> only the full window."""
    tail = MIN_TAIL - 1
    tokens, states = _arr(WINDOW_LEN + tail)
    chunks = list(chunk_read(tokens, states))
    assert len(chunks) == 1
    # The full window comes from the start of the input.
    np.testing.assert_array_equal(chunks[0][0], tokens[:WINDOW_LEN])


def test_window_plus_long_tail_keeps_padded_tail():
    """One full window + a tail >= min_tail -> two windows, second is padded."""
    tail = MIN_TAIL + 5
    tokens, states = _arr(WINDOW_LEN + tail)
    chunks = list(chunk_read(tokens, states))
    assert len(chunks) == 2
    # Full window
    np.testing.assert_array_equal(chunks[0][0], tokens[:WINDOW_LEN])
    np.testing.assert_array_equal(chunks[0][1], states[:WINDOW_LEN])
    # Padded tail
    pad_tokens, pad_states = chunks[1]
    assert pad_tokens.shape == (WINDOW_LEN,)
    assert pad_states.shape == (WINDOW_LEN,)
    np.testing.assert_array_equal(pad_tokens[:tail], tokens[WINDOW_LEN:])
    np.testing.assert_array_equal(pad_states[:tail], states[WINDOW_LEN:])
    # Padding region uses the default pad values.
    assert (pad_tokens[tail:] == DEFAULT_PAD_TOKEN).all()
    assert (pad_states[tail:] == DEFAULT_PAD_STATE).all()


def test_two_full_windows_no_tail():
    tokens, states = _arr(2 * WINDOW_LEN)
    chunks = list(chunk_read(tokens, states))
    assert len(chunks) == 2
    np.testing.assert_array_equal(chunks[0][0], tokens[:WINDOW_LEN])
    np.testing.assert_array_equal(chunks[1][0], tokens[WINDOW_LEN:])


def test_two_full_windows_plus_long_tail():
    tail = MIN_TAIL + 100
    tokens, states = _arr(2 * WINDOW_LEN + tail)
    chunks = list(chunk_read(tokens, states))
    assert len(chunks) == 3
    # Padded tail content
    pad_tokens, _ = chunks[2]
    np.testing.assert_array_equal(pad_tokens[:tail], tokens[2 * WINDOW_LEN:])
    assert (pad_tokens[tail:] == DEFAULT_PAD_TOKEN).all()


def test_realistic_long_read():
    """A realistic ~7400-token read should produce 14 windows of length 510 each
    plus a tail. 7400 / 510 = 14 with remainder 260 (>= MIN_TAIL=256), so 15 in total.
    """
    L = 7400
    tokens, states = _arr(L)
    chunks = list(chunk_read(tokens, states))
    n_full = L // WINDOW_LEN
    tail = L - n_full * WINDOW_LEN
    expected_total = n_full + (1 if tail >= MIN_TAIL else 0)
    assert len(chunks) == expected_total
    for t, s in chunks:
        assert t.shape == (WINDOW_LEN,)
        assert s.shape == (WINDOW_LEN,)


# ---- Window content correctness -----------------------------------------

def test_windows_are_non_overlapping_and_cover_input():
    """Concatenating window contents (excluding tail padding) reproduces input."""
    L = 3 * WINDOW_LEN + MIN_TAIL + 10
    tokens, states = _arr(L)
    chunks = list(chunk_read(tokens, states))
    # Reconstruct: first three windows are full, last is padded with tail prefix.
    reconstructed_tokens = np.concatenate([
        chunks[0][0], chunks[1][0], chunks[2][0],
        chunks[3][0][:MIN_TAIL + 10],
    ])
    np.testing.assert_array_equal(reconstructed_tokens, tokens)


# ---- Custom pad values --------------------------------------------------

def test_custom_pad_values_are_respected():
    tail = MIN_TAIL
    tokens, states = _arr(WINDOW_LEN + tail)
    chunks = list(chunk_read(
        tokens, states,
        pad_token=42, pad_state=3,
    ))
    assert len(chunks) == 2
    pad_tokens, pad_states = chunks[1]
    assert (pad_tokens[tail:] == 42).all()
    assert (pad_states[tail:] == 3).all()


# ---- Custom window/min_tail --------------------------------------------

def test_custom_window_and_min_tail():
    tokens, states = _arr(105)  # 100 -> two full + 5 tail
    chunks = list(chunk_read(tokens, states, window_len=50, min_tail=4))
    assert len(chunks) == 3  # 2 full + 1 padded tail of length 5
    assert chunks[0][0].shape == (50,)
    assert chunks[1][0].shape == (50,)
    assert chunks[2][0].shape == (50,)
    np.testing.assert_array_equal(chunks[2][0][:5], tokens[100:])


# ---- Validation ---------------------------------------------------------

def test_length_mismatch_raises():
    tokens = np.arange(10, dtype=np.int32)
    states = np.zeros(11, dtype=np.int8)
    with pytest.raises(ValueError, match="same length"):
        list(chunk_read(tokens, states))


def test_dtype_preserved():
    """The chunker should not alter input dtypes (writer handles dtype conversion)."""
    tokens = np.arange(WINDOW_LEN, dtype=np.int32)
    states = np.zeros(WINDOW_LEN, dtype=np.int8)
    [(t, s)] = list(chunk_read(tokens, states))
    assert t.dtype == np.int32
    assert s.dtype == np.int8