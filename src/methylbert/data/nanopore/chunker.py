"""
methylbert.data.nanopore.chunker

Slice featurized reads into fixed-length training windows.

A single nanopore read is typically 7-28 kbp long. After featurize_read produces
parallel (token_ids, methyl_states) arrays, we slice them into non-overlapping
windows of WINDOW_LEN tokens each. The tail of each read is dropped if it has
fewer than MIN_TAIL tokens; otherwise it is padded up to WINDOW_LEN.

The chunker is a pure function with no I/O. Output windows are uniform fixed-
length and ready to be appended to a binary shard.
"""

from typing import Iterator, Tuple

import numpy as np


WINDOW_LEN = 510
MIN_TAIL = 256

# Defaults match MethylVocab.pad_index (= 0) and the existing convention of
# state 2 (non-CpG) at SOS/EOS/pad positions in finetune dataset code.
DEFAULT_PAD_TOKEN = 0
DEFAULT_PAD_STATE = 2


def chunk_read(
    token_ids: np.ndarray,
    methyl_states: np.ndarray,
    window_len: int = WINDOW_LEN,
    min_tail: int = MIN_TAIL,
    pad_token: int = DEFAULT_PAD_TOKEN,
    pad_state: int = DEFAULT_PAD_STATE,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Slice (token_ids, methyl_states) into fixed-length non-overlapping windows.

    Each yielded pair is exactly window_len long. Full windows come first,
    sliced from the start of the input. If the tail (the leftover after the
    last full window) is at least min_tail tokens long, it is yielded as a
    final padded window; otherwise it is dropped.

    Parameters
    ----------
    token_ids : np.ndarray, shape (L,)
        DNA 3-mer token IDs, as produced by featurize_read.
    methyl_states : np.ndarray, shape (L,)
        Methylation states in {0, 1, 2, 3}, parallel to token_ids.
    window_len : int
        Window length in tokens. Default 510.
    min_tail : int
        Minimum length for the tail to be retained (and padded). Default 256.
    pad_token : int
        Pad value for the token track in the tail. Default 0
        (MethylVocab.pad_index).
    pad_state : int
        Pad value for the methylation track in the tail. Default 2 (non-CpG),
        matching the convention used by the existing finetune dataset code for
        SOS/EOS/pad positions.

    Yields
    ------
    (window_tokens, window_states) : tuple of np.ndarray
        Each shape (window_len,). All yielded windows are exactly window_len long.
        Tokens preserve the input dtype; states preserve the input dtype.

    Raises
    ------
    ValueError
        If token_ids and methyl_states have different lengths.
    """
    if len(token_ids) != len(methyl_states):
        raise ValueError(
            f"token_ids and methyl_states must have the same length; "
            f"got {len(token_ids)} and {len(methyl_states)}"
        )

    L = len(token_ids)
    if L == 0:
        return

    n_full = L // window_len
    for i in range(n_full):
        start = i * window_len
        end = start + window_len
        yield token_ids[start:end], methyl_states[start:end]

    tail_start = n_full * window_len
    tail_len = L - tail_start
    if tail_len >= min_tail:
        padded_tokens = np.full(window_len, pad_token, dtype=token_ids.dtype)
        padded_states = np.full(window_len, pad_state, dtype=methyl_states.dtype)
        padded_tokens[:tail_len] = token_ids[tail_start:]
        padded_states[:tail_len] = methyl_states[tail_start:]
        yield padded_tokens, padded_states