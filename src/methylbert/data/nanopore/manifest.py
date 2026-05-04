"""
methylbert.data.nanopore.manifest

Manifest schema and loader for the nanopore preprocessing pipeline.

The manifest is a TSV file listing one row per BAM with its per-BAM properties.
It is the source of truth for per-BAM configuration the featurizer/preprocessor
needs (most importantly the MM tag flag character) and a documentation record
of the corpus composition (reference build, basecaller version, mod codes,
cohort, sample label).

Required columns (tab-separated):

    bam_path             Absolute path to the BAM file. (Must be unique.)
    mm_flag              MM tag flag character: '.' (implicit canonical for
                         unlisted Cs) or '?' (explicit unknown). Determines how
                         the featurizer labels CpG Cs not listed in the MM tag.
    reference_build      e.g. "CHM13v2.0", "hg38". Informational; the
                         preprocessor uses read sequence not reference.
    basecaller_version   e.g. "dorado_0.5.3". Informational, useful for
                         documenting per-BAM ML calibration.
    mod_codes            Comma-separated MM tag mod codes present, e.g.
                         "C+m,C+h". Informational.
    cohort               Free-form cohort label, used later for balanced
                         sampling at training time (e.g. "HG008", "COLO829").
    sample_label         Short human-readable label, e.g.
                         "normal_duodenal", "tumor_pdac".
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List


REQUIRED_COLUMNS = [
    "bam_path",
    "mm_flag",
    "reference_build",
    "basecaller_version",
    "mod_codes",
    "cohort",
    "sample_label",
]

VALID_MM_FLAGS = (".", "?")


@dataclass
class ManifestEntry:
    bam_path: str
    mm_flag: str
    reference_build: str
    basecaller_version: str
    mod_codes: str
    cohort: str
    sample_label: str

    def __post_init__(self):
        if self.mm_flag not in VALID_MM_FLAGS:
            raise ValueError(
                f"mm_flag must be one of {VALID_MM_FLAGS}, got "
                f"{self.mm_flag!r} for BAM {self.bam_path}"
            )
        if not self.bam_path:
            raise ValueError("bam_path must not be empty")


def load_manifest(path) -> List[ManifestEntry]:
    """
    Load and validate a TSV manifest file.

    Parameters
    ----------
    path : str or Path
        Path to the TSV manifest.

    Returns
    -------
    list of ManifestEntry
        One entry per BAM, in file order.

    Raises
    ------
    FileNotFoundError
        If the manifest file does not exist.
    ValueError
        If the manifest is empty, missing required columns, or contains a
        duplicate bam_path or invalid mm_flag.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    entries: List[ManifestEntry] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Empty manifest file: {path}")
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing)}. "
                f"Found columns: {reader.fieldnames}"
            )
        # Row indices reported in errors are 1-indexed and include the header
        # at line 1; data rows therefore start at line 2.
        for line_num, row in enumerate(reader, start=2):
            try:
                entry = ManifestEntry(**{k: row[k] for k in REQUIRED_COLUMNS})
            except (TypeError, ValueError) as e:
                raise ValueError(f"Manifest line {line_num}: {e}") from e
            entries.append(entry)

    if not entries:
        raise ValueError(f"Manifest contains no data rows: {path}")

    # Reject duplicate BAM paths
    seen = set()
    for entry in entries:
        if entry.bam_path in seen:
            raise ValueError(
                f"Duplicate bam_path in manifest: {entry.bam_path}"
            )
        seen.add(entry.bam_path)

    return entries