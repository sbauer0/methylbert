import argparse
import random
import numpy as np
import re
import os
import multiprocessing as mp
from functools import partial
import pysam
from Bio import SeqIO
from tqdm.auto import tqdm
from .bam import process_bismark_read
import json

# NCBI-style accession to chr name mapping for hg38
NCBI_TO_CHR = {
    "NC_000001.11": "chr1",  "NC_000002.12": "chr2",
    "NC_000003.12": "chr3",  "NC_000004.12": "chr4",
    "NC_000005.10": "chr5",  "NC_000006.12": "chr6",
    "NC_000007.14": "chr7",  "NC_000008.11": "chr8",
    "NC_000009.12": "chr9",  "NC_000010.11": "chr10",
    "NC_000011.10": "chr11", "NC_000012.12": "chr12",
    "NC_000013.11": "chr13", "NC_000014.9":  "chr14",
    "NC_000015.10": "chr15", "NC_000016.10": "chr16",
    "NC_000017.11": "chr17", "NC_000018.12": "chr18",
    "NC_000019.10": "chr19", "NC_000020.11": "chr20",
    "NC_000021.9":  "chr21", "NC_000022.11": "chr22",
    "NC_000023.11": "chrX",  "NC_000024.10": "chrY"
}

def _kmers_5base_tokens(seq: str, vocab, k: int = 3):
    '''
        Convert a methylation-annotated reference sequence directly into a list
        of token indices (int), skipping the intermediate string representation.

        Bismark conventions used:
            Z = methylated CpG cytosine   -> substituted with M
            z = unmethylated CpG cytosine -> kept as C
            h, H, x, X                   -> non-CpG methylation, kept as C

        seq : str
            Methylation-annotated reference sequence.
        vocab : MethylVocab
            Vocabulary object for 3-mer -> token index lookup.
        k : int
            k for k-mer tokenisation. Must be odd. Default: 3.
    '''
    if k % 2 == 0:
        raise ValueError(f"k must be odd. Given: {k}")

    # Substitute methylated CpG cytosine (Z) -> M, everything else -> C
    seq = re.sub("[zZhHxX]", lambda m: "M" if m.group() == "Z" else "C", seq)

    stoi = vocab.stoi
    unk = vocab.unk_index
    return [stoi.get(seq[i:i + k], unk) for i in range(len(seq) - k)]

"""def _process_read_pretrain_binary(read, dict_ref, vocab, k, seq_len, min_seq_len,
                                  methyl_caller="bismark"):
    if read.query_alignment_length < min_seq_len:
        if not hasattr(_process_read_pretrain_binary, '_dbg'):
            _process_read_pretrain_binary._dbg = True
            print(f"DEBUG reject (len): {read.query_alignment_length} < {min_seq_len}")
        return None

    chromo = read.reference_name
    if chromo not in dict_ref:
        if not hasattr(_process_read_pretrain_binary, '_dbg_chr'):
            _process_read_pretrain_binary._dbg_chr = True
            print(f"DEBUG reject (chr): {chromo} not in dict_ref. Sample keys: {list(dict_ref.keys())[:5]}")
        return None

    ref_seq = dict_ref[chromo][read.pos:(read.pos + read.query_alignment_length)].upper()

    if methyl_caller == "bismark":
        annotated_seq = process_bismark_read(ref_seq, read)
    else:
        raise ValueError(f"Unsupported methyl_caller: {methyl_caller}")

    if annotated_seq is None:
        if read.is_duplicate:
            if not hasattr(_process_read_pretrain_binary, '_dbg_dup'):
                _process_read_pretrain_binary._dbg_dup = True
                print(f"DEBUG reject (dup): duplicate")
            return None
        annotated_seq = ref_seq

    tokens = _kmers_5base_tokens(annotated_seq, vocab, k=k)

    if len(tokens) == 0:
        if not hasattr(_process_read_pretrain_binary, '_dbg_empty'):
            _process_read_pretrain_binary._dbg_empty = True
            print(f"DEBUG reject (empty tokens): seq_len={len(annotated_seq)}")
        return None

    if not hasattr(_process_read_pretrain_binary, '_dbg_ok'):
        _process_read_pretrain_binary._dbg_ok = True
        print(f"DEBUG accept: n_tokens={len(tokens)}, first_few={tokens[:5]}")

    if len(tokens) >= seq_len:
        tokens = tokens[:seq_len]
    else:
        tokens = tokens + [vocab.pad_index] * (seq_len - len(tokens))

    return np.array(tokens, dtype=np.int16)"""
def _process_read_pretrain_binary(read, dict_ref, vocab, k, seq_len, min_seq_len,
                                  methyl_caller="bismark"):
    '''
        Process a single pysam read into a fixed-length int16 token array of
        shape (seq_len,). Sequences shorter than seq_len tokens are padded with
        vocab.pad_index; sequences longer are truncated.

        Reads are discarded if:
        - Their aligned length is strictly less than min_seq_len
        - Their reference_name is not in dict_ref

        read : pysam.AlignedSegment
        dict_ref : dict
        vocab : MethylVocab
        k : int
        seq_len : int
            Fixed number of tokens per output row.
        min_seq_len : int
            Minimum aligned read length in base pairs. Reads shorter than this
            are discarded.
        methyl_caller : str
    '''
    if read.query_alignment_length < min_seq_len:
        return None

    chromo = read.reference_name
    if chromo not in dict_ref:
        return None

    ref_seq = dict_ref[chromo][read.pos:(read.pos + read.query_alignment_length)].upper()

    if methyl_caller == "bismark":
        annotated_seq = process_bismark_read(ref_seq, read)
    else:
        raise ValueError(f"Unsupported methyl_caller: {methyl_caller}. Only 'bismark' is supported.")

    if annotated_seq is None:
        if read.is_duplicate:
            return None
        annotated_seq = ref_seq

    tokens = _kmers_5base_tokens(annotated_seq, vocab, k=k)

    if len(tokens) == 0:
        return None

    # Truncate or pad to exactly seq_len
    if len(tokens) >= seq_len:
        tokens = tokens[:seq_len]
    else:
        tokens = tokens + [vocab.pad_index] * (seq_len - len(tokens))

    return np.array(tokens, dtype=np.int16)


def _process_bam_file_binary(bam_path, dict_ref, vocab, k, seq_len, min_seq_len,
                             output_dir, methyl_caller="bismark", max_reads=None):
    '''
        Process all reads in a single BAM file, writing tokenized sequences
        directly to a binary .bin file plus a .json sidecar with metadata.

        bam_path : str
        dict_ref : dict
        vocab : MethylVocab
        k : int
        seq_len : int
        output_dir : str
            Directory where output files are written.
        methyl_caller : str
        max_reads : int
    '''
    bam_path = bam_path.strip()
    bam_name = os.path.splitext(os.path.basename(bam_path))[0]
    f_bin = os.path.join(output_dir, f"{bam_name}.bin")
    f_json = os.path.join(output_dir, f"{bam_name}.json")

    # Skip if already processed (resumability)
    if os.path.exists(f_bin) and os.path.exists(f_json):
        with open(f_json, "r") as f:
            meta = json.load(f)
        print(f"  Skipping {bam_name}: already processed ({meta['n_reads']:,} reads)")
        return f_bin, meta["n_reads"]

    n_kept = 0
    try:
        aln = pysam.AlignmentFile(bam_path, "rb")
        with open(f_bin, "wb") as f_out:
            for i, read in enumerate(aln.fetch(until_eof=True)):
                if max_reads is not None and i >= max_reads:
                    print(f"  [{bam_name}] Reached max_reads limit ({max_reads}), stopping.")
                    break
                if i % 500000 == 0 and i > 0:
                    print(f"  [{bam_name}] Processed {i:,} reads, kept {n_kept:,}...", flush=True)
                if read.is_unmapped:
                    continue
                tokens = _process_read_pretrain_binary(
                    read, dict_ref, vocab, k, seq_len, min_seq_len, methyl_caller
                )
                if tokens is not None:
                    f_out.write(tokens.tobytes())
                    n_kept += 1
        aln.close()
    except Exception as e:
        print(f"Warning: failed to process {bam_path}: {e}")
        return None, 0

    # Write sidecar metadata
    meta = {
        "n_reads": n_kept,
        "seq_len": seq_len,
        "k": k,
        "vocab_size": len(vocab),
        "source_bam": bam_path,
    }
    with open(f_json, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [{bam_name}] Done. Kept {n_kept:,} reads -> {f_bin}")
    return f_bin, n_kept


def pretrain_data_preprocess_5base_binary(
    f_ref,
    sc_dataset,
    vocab,
    seq_len=150,
    min_seq_len=150,
    k=3,
    n_cores=1,
    methyl_caller="bismark",
    output_dir=None,
    max_reads=None
):
    '''
        Generate pretraining data using the 5-base alphabet (A, C, T, G, M) from
        BAM files containing bisulfite-sequenced reads. Writes one int16 binary
        file per BAM, along with a JSON sidecar containing shape metadata.
        Output is directly consumable by MethylBertPretrainDatasetBinary via
        np.memmap, bypassing the text-based round-trip entirely.

        f_ref : str
            Path to the reference genome FASTA file (NCBI GRCh38 format).
        sc_dataset : str
            Path to a plain text file listing BAM file paths, one per line.
            Optionally tab-separated with a second column for cell-type labels
            (labels are ignored for pretraining).
        vocab : MethylVocab
            Vocabulary object for tokenisation. Must be the same vocab used
            for model training.
        seq_len : int
            Fixed number of tokens per output row. Reads shorter than
            seq_len + k bases are discarded. Default: 150.
        k : int
            k for k-mer tokenisation. Must be odd. Default: 3.
        n_cores : int
            Number of cores for parallel BAM processing. One BAM per core.
            Default: 1.
        methyl_caller : str
            Methylation caller used to generate the BAM files. Currently only
            'bismark' is supported. Default: 'bismark'.
        output_dir : str
            Directory where .bin and .json files are written. If None, a
            directory named after sc_dataset with a '_5base_pretrain_bin'
            suffix is created.
        max_reads : int
            Maximum number of reads to iterate over per BAM. Useful for
            testing. If None, all reads are processed. Default: None.
    '''
    # --- Output directory ---
    if output_dir is None:
        output_dir = os.path.splitext(sc_dataset)[0] + "_5base_pretrain_bin"
    os.makedirs(output_dir, exist_ok=True)

    # --- Load reference genome ---
    print("Loading reference genome...")
    dict_ref = {}
    for record in SeqIO.parse(f_ref, "fasta"):
        record_id = str(record.id)
        if record_id in NCBI_TO_CHR:
            seq = str(record.seq.upper())
            # Populate with both naming conventions so BAMs from any source match:
            dict_ref[record_id] = seq                                    # NCBI: NC_000001.11
            dict_ref[NCBI_TO_CHR[record_id].replace("chr", "")] = seq    # bare: 1
    print(f"Loaded {len(dict_ref) // 2} chromosomes (keyed by both NCBI and bare-numeric names).")
    print(f"Sample keys: {list(dict_ref.keys())[:6]}")

    # --- Read BAM file list ---
    with open(sc_dataset, "r") as f:
        bam_paths = [line.strip().split("\t")[0]
                     for line in f if line.strip()]
    print(f"Found {len(bam_paths)} BAM file(s).")

    # --- Validate naming convention ---
    _check_bam_naming_consistency(bam_paths, dict_ref)

    # --- Process BAM files ---
    process_fn = partial(
        _process_bam_file_binary,
        dict_ref=dict_ref,
        vocab=vocab,
        k=k,
        seq_len=seq_len,
        min_seq_len=min_seq_len,
        output_dir=output_dir,
        methyl_caller=methyl_caller,
        max_reads=max_reads,
    )

    results = []
    if n_cores > 1:
        with mp.Pool(n_cores) as pool:
            results = list(tqdm(
                pool.imap(process_fn, bam_paths),
                total=len(bam_paths),
                desc="Processing BAM files"
            ))
    else:
        for bam_path in tqdm(bam_paths, desc="Processing BAM files"):
            results.append(process_fn(bam_path))

    # --- Summary ---
    total_reads = sum(n for _, n in results if n is not None)
    print(f"\nTotal reads across all BAMs: {total_reads:,}")
    print(f"Output directory: {output_dir}")

    return output_dir
     
def _kmers_5base(seq: str, k: int = 3):
    """
    Convert a methylation-annotated reference sequence into k-mer tokens
    using the 5-base alphabet (A, C, T, G, M).

    Bismark conventions used:
        Z = methylated CpG cytosine   -> substituted with M
        z = unmethylated CpG cytosine -> kept as C
        h, H, x, X                   -> non-CpG methylation, kept as C

    Returns a single list of k-mer token strings (no separate methylation track).
    """
    if k % 2 == 0:
        raise ValueError(f"k must be odd. Given: {k}")

    # Substitute methylated CpG cytosine (Z) -> M, everything else -> C
    seq = re.sub("[zZhHxX]", lambda m: "M" if m.group() == "Z" else "C", seq)

    tokens = []
    for i in range(len(seq) - k):
        tokens.append(seq[i:i + k])

    return tokens




def _process_read_pretrain_5base(read, dict_ref, k, min_seq_len, methyl_caller="bismark"):
    '''
        Process a single pysam read into a 5-base k-mer token string.
        Returns a space-separated string of k-mers, or None if the read is discarded.

        read : pysam.AlignedSegment
            A single aligned read from a BAM file.
        dict_ref : dict
            Reference genome as a dictionary mapping chromosome name to
            its full DNA sequence string (uppercase).
        k : int
            k for k-mer tokenisation. Must be odd.
        min_seq_len : int
            Minimum read length in base pairs. Reads strictly shorter than
            this value are discarded.
        methyl_caller : str
            Methylation caller used to generate the BAM. Only 'bismark'
            is supported. Determines how the XM tag is interpreted to
            reconstruct methylation-annotated reference sequences.
    '''

    # Discard reads shorter than min_seq_len
    if read.query_alignment_length < min_seq_len:
        return None

    chromo = read.reference_name
    if chromo not in dict_ref:
        return None

    ref_seq = dict_ref[chromo][read.pos:(read.pos + read.query_alignment_length)].upper()

    if methyl_caller == "bismark":
        annotated_seq = process_bismark_read(ref_seq, read)
    else:
        raise ValueError(f"Unsupported methyl_caller: {methyl_caller}. Only 'bismark' is supported.")

    # process_bismark_read returns None for duplicates, missing XM tag,
    # or reads with no CpG sites. In those cases we fall back to the plain
    # reference sequence — these reads will tokenize as normal A/C/T/G 3-mers
    # with no M substitution, which preserves non-CpG sequence context during
    # pretraining rather than discarding it entirely.
    if annotated_seq is None:
        annotated_seq = ref_seq

    tokens = _kmers_5base(annotated_seq, k=k)

    if len(tokens) == 0:
        return None

    return " ".join(tokens)


def _process_bam_file(bam_path, dict_ref, k, min_seq_len, methyl_caller="bismark", max_reads=None):
    '''
        Process all reads in a single BAM file.
        Returns a list of space-separated k-mer strings (one per valid read).

        bam_path : str
            Path to the BAM file to process.
        dict_ref : dict
            Reference genome as a dictionary mapping chromosome name to
            its full DNA sequence string (uppercase).
        k : int
            k for k-mer tokenisation. Must be odd.
        min_seq_len : int
            Minimum read length in base pairs. Reads strictly shorter than
            this value are discarded.
        methyl_caller : str
            Methylation caller used to generate the BAM. Only 'bismark'
            is supported. Default: 'bismark'.
        max_reads : int
            Maximum number of reads to iterate over. Processing stops once
            this limit is reached. If None, all reads are processed. Default: None.
    '''
    results = []
    try:
        aln = pysam.AlignmentFile(bam_path.strip(), "rb")
        for i, read in enumerate(aln.fetch(until_eof=True)):
            if max_reads is not None and i >= max_reads:
                print(f"  Reached max_reads limit ({max_reads}), stopping.")
                break
            if i % 1000000 == 0:
                print(f"  Processed {i:,} reads, kept {len(results):,}...", flush=True)
            if read.is_unmapped:
                continue
            line = _process_read_pretrain_5base(read, dict_ref, k, min_seq_len, methyl_caller)
            if line is not None:
                results.append(line)
        aln.close()
    except Exception as e:
        print(f"Warning: failed to process {bam_path}: {e}")

    return results


def pretrain_data_preprocess_5base(
    f_ref,
    sc_dataset,
    min_seq_len=150,
    k=3,
    n_cores=1,
    methyl_caller="bismark",
    f_output=None,
    max_reads=None
):
    '''
        Generate pretraining data using the 5-base alphabet (A, C, T, G, M) from
        BAM files containing bisulfite-sequenced reads. Output format mirrors
        pretrain_data_preprocess: one line per sequence, space-separated k-mer
        tokens, directly compatible with MethylBertPretrainDataset.

        f_ref : str
            Path to the reference genome FASTA file (NCBI GRCh38 format).
        sc_dataset : str
            Path to a plain text file listing BAM file paths, one per line.
            Optionally tab-separated with a second column for cell-type labels
            (labels are ignored for pretraining).
        min_seq_len : int
            Minimum read length in base pairs. Reads shorter than this are
            discarded. Default: 150.
        k : int
            k for k-mer tokenisation. Must be odd. Default: 3.
        n_cores : int
            Number of cores for parallel BAM processing. Default: 1.
        methyl_caller : str
            Methylation caller used to generate the BAM files. Currently only
            'bismark' is supported. Default: 'bismark'.
        f_output : str
            Path to the output text file. If None, the output is written next
            to sc_dataset with a '_5base_pretrain.txt' suffix.
        max_reads : int
            Maximum number of valid reads to collect per BAM file. Useful for
            testing. If None, all reads are processed. Default: None.
    '''

    # --- Output path ---
    if f_output is None:
        f_output = os.path.splitext(sc_dataset)[0] + "_5base_pretrain.txt"
    os.makedirs(os.path.dirname(f_output), exist_ok=True)

    # --- Load reference genome ---
    print("Loading reference genome...")
    dict_ref = {}
    for record in SeqIO.parse(f_ref, "fasta"):
        record_id = str(record.id)
        if record_id in NCBI_TO_CHR:
            # Strip "chr" prefix to match BAMs using bare numeric names
            chr_name = NCBI_TO_CHR[record_id].replace("chr", "")
            dict_ref[chr_name] = str(record.seq.upper())
    print(f"Loaded {len(dict_ref)} chromosomes.")
    print(f"Keys: {list(dict_ref.keys())[:5]}")  # sanity check

    # --- Read BAM file list ---
    with open(sc_dataset, "r") as f:
        bam_paths = [line.strip().split("\t")[0]  # ignore optional cell-type column
                     for line in f if line.strip()]
    print(f"Found {len(bam_paths)} BAM file(s).")

    # --- Validate naming convention ---
    _check_bam_naming_consistency(bam_paths, dict_ref)

    # --- Process BAM files ---
    process_fn = partial(
        _process_bam_file,
        dict_ref=dict_ref,
        k=k,
        min_seq_len=min_seq_len,
        methyl_caller=methyl_caller,
        max_reads=max_reads
    )

    all_sequences = []
    if n_cores > 1:
        with mp.Pool(n_cores) as pool:
            results = list(tqdm(
                pool.imap(process_fn, bam_paths),
                total=len(bam_paths),
                desc="Processing BAM files"
            ))
        for r in results:
            all_sequences.extend(r)
    else:
        for bam_path in tqdm(bam_paths, desc="Processing BAM files"):
            all_sequences.extend(process_fn(bam_path))

    print(f"Total sequences written: {len(all_sequences)}")

    # --- Write output ---
    with open(f_output, "w") as f_out:
        for seq in all_sequences:
            f_out.write(seq + "\n")

    print(f"Output written to: {f_output}")
    return f_output


def _check_bam_naming_consistency(bam_paths, dict_ref):
    '''
        Verify that all BAM files use reference names consistent with dict_ref.
        Checks by sampling the first mapped read from each BAM and comparing
        its reference_name against the keys of dict_ref.

        bam_paths : list[str]
            List of paths to BAM files.
        dict_ref : dict
            Reference genome dictionary whose keys are the expected chromosome
            identifiers.

        Raises
        ------
        ValueError
            If any BAM file uses reference names not present in dict_ref, or
            if a BAM file contains no mapped reads to check.
    '''
    ref_keys = set(dict_ref.keys())
    mismatches = []

    for bam_path in bam_paths:
        bam_path = bam_path.strip()
        try:
            aln = pysam.AlignmentFile(bam_path, "rb")
        except Exception as e:
            raise ValueError(f"Could not open BAM file {bam_path}: {e}")

        sample_name = None
        for read in aln.fetch(until_eof=True):
            if not read.is_unmapped:
                sample_name = read.reference_name
                break
        aln.close()

        if sample_name is None:
            raise ValueError(f"BAM file {bam_path} contains no mapped reads.")

        if sample_name not in ref_keys:
            mismatches.append((bam_path, sample_name))

    if mismatches:
        msg_lines = [
            "Naming convention mismatch between reference genome and BAM file(s).",
            f"Reference dict_ref uses keys like: {list(ref_keys)[:5]}",
            "Mismatched BAM files:"
        ]
        for path, name in mismatches:
            msg_lines.append(f"  - {path} uses reference name '{name}'")
        raise ValueError("\n".join(msg_lines))

    print(f"Naming convention check passed for {len(bam_paths)} BAM file(s).")


def _kmers(original_string, kmer=3):
    sentence = ""
    #original_string = original_string.replace("\n", "")
    i = 0
    while i < len(original_string)-kmer:
        sentence += original_string[i:i+kmer] + " "
        i += 1
    
    return sentence[:-1].strip("\"")


def pretrain_data_preprocess(f_ref, k=3, seq_len=510, f_output=None):
    '''
        Generate N bp length k-mers sequence from reference genome as pretrain data

        f_ref : str
            path to the reference fasta file
        k : int
            Number for k-mers (default=3)
        seq_len: int
            Base-pair length of generated sequences (default=510)
        f_output : str
            path to the output file, an appropriate name 
            will be automatically assigned if not given

    '''

    fp_ref = open(f_ref, "r")
    if f_output == None:
        f_output = f_ref + "_%dmers.txt"%(k)
    
    fp_out = open(f_output, "w")
    line = fp_ref.readline().strip().upper()
    cur_line="" # keeping an incomplete N bp line
    collect_data=False
    valid_chromosomes = ["CHR"+str(i) for i in range(22)]
    valid_chromosomes += ["CHRX", "CHRY"]
    
    while line:
        n_missing = line.count("N")

        if n_missing > 0:
            # Missing DNA base in the line -> reset the line 
            #line = fp_ref.readline().strip().upper()
            line=fp_ref.readline().strip().upper()
            cur_line=""
            continue
        elif ( line.count(">") > 0 ):
            # New chromosome or there are some missing bases at the middle
            # We restart a 510 bp piece
            chromosome = line.split(">")[1]
            collect_data = chromosome in valid_chromosomes
            if collect_data:
                print("Collect sequences in %s"%(chromosome))
            line = fp_ref.readline().strip().upper()
            cur_line = "" 
            continue

        if collect_data:
            cur_line += line
            line_length = len(cur_line)
            if line_length >= seq_len:
                new_line = cur_line[:seq_len]
                cur_line = cur_line[seq_len:]
                sentence = _kmers(new_line, kmer=k)
                fp_out.write(sentence + "\n")

        # get a new line 
        line = fp_ref.readline().strip().upper()
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kmer",
        default=1,
        type=int,
        help="K-mer",
    )
    parser.add_argument(
        "--length",
        default=10000,
        type=int,
        help="Length of the sampled sequence",
    )
    parser.add_argument(
        "--file_path",
        default=None,
        type=str,
        help="The path of the file to be processed",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="The path of the processed data",
    )
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()
    pretrain_data_preprocess(args.file_path, k=args.kmer, 
                             seq_len=args.length, f_output=args.output_path)

