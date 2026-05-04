"""
methylbert.data.nanopore.shards

Binary shard writer for the nanopore preprocessing pipeline.

For each BAM, three files are written into the output directory:

    <bam_basename>.tokens.bin   flat int16 array of shape (n_rows, window_len)
    <bam_basename>.states.bin   flat int8  array of shape (n_rows, window_len)
    <bam_basename>.json         sidecar with metadata + manifest entry

Tokens and states are stored in two separate files so each can be a uniform-
dtype memory-mapped numpy array on the read side. Row i in `tokens` and row i
in `states` are the parallel arrays for the same training window.

The writer is:
    - resumable: shard_exists(...) returns True for any BAM with a complete,
      consistent set of three files; the preprocessor driver uses this to
      skip BAMs that are already done.
    - atomic: data is written to .tmp paths first; the .tokens.bin /
      .states.bin / .json files are only created via os.replace once all
      three are fully written. A crashed run never leaves valid-looking
      partial files.
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np


SHARD_VERSION = 1   # bump if the on-disk format changes


class ShardWriter:
    """
    Append-only binary writer for one BAM's training windows.

    Usage:
        with ShardWriter(output_dir, bam_basename, window_len,
                         manifest_entry, vocab_size) as w:
            for tokens, states in chunked_windows:
                w.append(tokens, states)

    On normal exit, three files appear atomically in output_dir. On exception,
    the temporary files are cleaned up and no shard files are produced.
    """

    def __init__(self, output_dir, bam_basename, window_len,
                 manifest_entry, vocab_size):
        self.output_dir = Path(output_dir)
        self.bam_basename = bam_basename
        self.window_len = int(window_len)
        self.manifest_entry = manifest_entry
        self.vocab_size = int(vocab_size)

        self._tokens_path = self.output_dir / f"{bam_basename}.tokens.bin"
        self._states_path = self.output_dir / f"{bam_basename}.states.bin"
        self._json_path = self.output_dir / f"{bam_basename}.json"

        # Use distinct .tmp paths so we can rename atomically on success.
        self._tokens_tmp = self.output_dir / f"{bam_basename}.tokens.bin.tmp"
        self._states_tmp = self.output_dir / f"{bam_basename}.states.bin.tmp"
        self._json_tmp = self.output_dir / f"{bam_basename}.json.tmp"

        self._tokens_fp = None
        self._states_fp = None
        self._n_rows = 0
        self._closed = False

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Clear any leftovers from a previous crashed run.
        for p in (self._tokens_tmp, self._states_tmp, self._json_tmp):
            if p.exists():
                p.unlink()
        self._tokens_fp = open(self._tokens_tmp, "wb")
        self._states_fp = open(self._states_tmp, "wb")
        return self

    def append(self, tokens: np.ndarray, states: np.ndarray) -> None:
        """Append one row (one training window)."""
        if self._closed:
            raise RuntimeError("ShardWriter is closed")
        if len(tokens) != self.window_len or len(states) != self.window_len:
            raise ValueError(
                f"Window length mismatch: expected {self.window_len}, "
                f"got tokens={len(tokens)}, states={len(states)}"
            )
        if tokens.dtype != np.int16:
            tokens = tokens.astype(np.int16)
        if states.dtype != np.int8:
            states = states.astype(np.int8)
        self._tokens_fp.write(tokens.tobytes())
        self._states_fp.write(states.tobytes())
        self._n_rows += 1

    @property
    def n_rows(self) -> int:
        return self._n_rows

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._closed:
            return False
        self._closed = True

        # Always close the temp file handles, even on error.
        for fp in (self._tokens_fp, self._states_fp):
            try:
                fp.close()
            except Exception:
                pass

        if exc_type is not None:
            # Roll back: drop the temp files. Don't suppress the exception.
            for p in (self._tokens_tmp, self._states_tmp, self._json_tmp):
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            return False

        # Write the JSON sidecar to its .tmp first.
        meta = {
            "shard_version": SHARD_VERSION,
            "n_rows": self._n_rows,
            "window_len": self.window_len,
            "vocab_size": self.vocab_size,
            "tokens_dtype": "int16",
            "states_dtype": "int8",
            "tokens_file": self._tokens_path.name,
            "states_file": self._states_path.name,
            "manifest_entry": asdict(self.manifest_entry),
        }
        with open(self._json_tmp, "w") as f:
            json.dump(meta, f, indent=2)

        # Atomic publish.
        os.replace(self._tokens_tmp, self._tokens_path)
        os.replace(self._states_tmp, self._states_path)
        os.replace(self._json_tmp, self._json_path)
        return False


def shard_exists(output_dir, bam_basename) -> bool:
    """
    Return True iff a complete, internally consistent shard exists for this BAM.

    Checks that:
      - all three files exist
      - the JSON has shard_version == SHARD_VERSION
      - the .bin file sizes match what the JSON metadata implies

    This is the predicate the preprocessor driver uses to decide whether to
    skip a BAM during a resumed run.
    """
    output_dir = Path(output_dir)
    json_path = output_dir / f"{bam_basename}.json"
    tokens_path = output_dir / f"{bam_basename}.tokens.bin"
    states_path = output_dir / f"{bam_basename}.states.bin"

    if not (json_path.exists() and tokens_path.exists() and states_path.exists()):
        return False

    try:
        with open(json_path) as f:
            meta = json.load(f)
        if meta.get("shard_version") != SHARD_VERSION:
            return False
        n_rows = int(meta["n_rows"])
        window_len = int(meta["window_len"])
        # int16 = 2 bytes, int8 = 1 byte
        if tokens_path.stat().st_size != n_rows * window_len * 2:
            return False
        if states_path.stat().st_size != n_rows * window_len * 1:
            return False
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        return False

    return True