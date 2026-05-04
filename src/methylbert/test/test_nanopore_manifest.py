"""
Tests for methylbert.data.nanopore.manifest.
"""

import pytest

from methylbert.data.nanopore.manifest import (
    ManifestEntry,
    load_manifest,
    REQUIRED_COLUMNS,
)


GOOD_HEADER = "\t".join(REQUIRED_COLUMNS)


def _row(bam_path="/data/A.bam", mm_flag=".", reference_build="hg38",
         basecaller_version="dorado_0.5.3", mod_codes="C+m,C+h",
         cohort="A", sample_label="sample_a"):
    return "\t".join([
        bam_path, mm_flag, reference_build, basecaller_version,
        mod_codes, cohort, sample_label,
    ])


# ---- Happy path ---------------------------------------------------------

def test_load_valid_manifest(tmp_path):
    p = tmp_path / "m.tsv"
    p.write_text(GOOD_HEADER + "\n" + _row() + "\n" +
                 _row(bam_path="/data/B.bam", mm_flag="?", cohort="B",
                      sample_label="sample_b") + "\n")
    entries = load_manifest(p)
    assert len(entries) == 2
    assert entries[0].bam_path == "/data/A.bam"
    assert entries[0].mm_flag == "."
    assert entries[1].bam_path == "/data/B.bam"
    assert entries[1].mm_flag == "?"
    assert entries[1].cohort == "B"


def test_manifest_entry_fields_are_accessible():
    e = ManifestEntry(
        bam_path="/x.bam", mm_flag=".", reference_build="hg38",
        basecaller_version="dorado_0.5.3", mod_codes="C+m,C+h",
        cohort="X", sample_label="x",
    )
    assert e.bam_path == "/x.bam"
    assert e.basecaller_version == "dorado_0.5.3"


# ---- Validation ---------------------------------------------------------

def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope.tsv")


def test_empty_file_raises(tmp_path):
    p = tmp_path / "m.tsv"
    p.write_text("")
    with pytest.raises(ValueError, match="Empty manifest"):
        load_manifest(p)


def test_header_only_raises(tmp_path):
    p = tmp_path / "m.tsv"
    p.write_text(GOOD_HEADER + "\n")
    with pytest.raises(ValueError, match="no data rows"):
        load_manifest(p)


def test_missing_column_raises(tmp_path):
    bad_header = "\t".join(c for c in REQUIRED_COLUMNS if c != "mm_flag")
    p = tmp_path / "m.tsv"
    p.write_text(bad_header + "\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_manifest(p)


def test_invalid_mm_flag_raises(tmp_path):
    p = tmp_path / "m.tsv"
    p.write_text(GOOD_HEADER + "\n" + _row(mm_flag="!") + "\n")
    with pytest.raises(ValueError, match="line 2"):
        load_manifest(p)


def test_invalid_mm_flag_in_dataclass_directly():
    with pytest.raises(ValueError, match="mm_flag"):
        ManifestEntry(
            bam_path="/x.bam", mm_flag="x", reference_build="hg38",
            basecaller_version="d", mod_codes="C+m",
            cohort="X", sample_label="x",
        )


def test_empty_bam_path_raises():
    with pytest.raises(ValueError, match="bam_path"):
        ManifestEntry(
            bam_path="", mm_flag=".", reference_build="hg38",
            basecaller_version="d", mod_codes="C+m",
            cohort="X", sample_label="x",
        )


def test_duplicate_bam_path_raises(tmp_path):
    p = tmp_path / "m.tsv"
    p.write_text(
        GOOD_HEADER + "\n" +
        _row(bam_path="/data/A.bam") + "\n" +
        _row(bam_path="/data/A.bam") + "\n"
    )
    with pytest.raises(ValueError, match="Duplicate bam_path"):
        load_manifest(p)


# ---- Real-corpus shape: 7 BAMs from our actual data --------------------

def test_real_corpus_layout(tmp_path):
    """Sanity check: write a manifest matching the actual 7-BAM corpus and
    confirm we can load it back with the right cohort labels."""
    p = tmp_path / "m.tsv"
    rows = [
        GOOD_HEADER,
        _row(bam_path="/data/HG008-N-D.bam", mm_flag=".",
             reference_build="CHM13v2.0", basecaller_version="dorado_0.5.3",
             cohort="HG008", sample_label="normal_duodenal"),
        _row(bam_path="/data/HG008-N-P.bam", mm_flag=".",
             reference_build="CHM13v2.0", basecaller_version="dorado_0.5.3",
             cohort="HG008", sample_label="normal_pancreatic"),
        _row(bam_path="/data/HG008-T.bam", mm_flag="?",
             reference_build="CHM13v2.0", basecaller_version="dorado_0.3.4",
             cohort="HG008", sample_label="tumor_pdac"),
        _row(bam_path="/data/PAU59949.bam", mm_flag="?",
             reference_build="hg38", basecaller_version="dorado_0.5.2",
             cohort="COLO829", sample_label="cancer"),
        _row(bam_path="/data/PAU61426.bam", mm_flag="?",
             reference_build="hg38", basecaller_version="dorado_0.5.2",
             cohort="COLO829", sample_label="cancer"),
        _row(bam_path="/data/PAU59807.bam", mm_flag="?",
             reference_build="hg38", basecaller_version="dorado_0.5.2",
             cohort="COLO829BL", sample_label="normal_blood"),
        _row(bam_path="/data/PAU61427.bam", mm_flag="?",
             reference_build="hg38", basecaller_version="dorado_0.5.2",
             cohort="COLO829BL", sample_label="normal_blood"),
    ]
    p.write_text("\n".join(rows) + "\n")
    entries = load_manifest(p)
    assert len(entries) == 7
    cohorts = [e.cohort for e in entries]
    assert cohorts.count("HG008") == 3
    assert cohorts.count("COLO829") == 2
    assert cohorts.count("COLO829BL") == 2
    flags = [e.mm_flag for e in entries]
    assert flags.count(".") == 2  # HG008 normals
    assert flags.count("?") == 5  # HG008-T + 4x COLO829*