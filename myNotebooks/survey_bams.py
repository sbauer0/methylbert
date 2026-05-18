#!/usr/bin/env python3
"""
BAM survey for nanopore methylation pretraining preprocessor design.

For each BAM, reports:
- Header info: @SQ count + sample (reference build), @PG entries (basecaller version),
  @RG entries (sample/chemistry).
- A sample of primary mapped reads with MM/ML tag analysis:
    * Modification codes present and their flag character ('?' / '.' / implicit)
    * ML probability distribution (per mod code where attributable, plus aggregate)
    * Read length, MAPQ, strand distribution.
- A few example reads showing raw MM/ML content for eyeballing context (CpG-only
  vs all-context, etc.).

Usage:
    python survey_bams.py BAM1 [BAM2 ...] [--n-reads N] [--out FILE.json]

Dependencies: pysam, numpy.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pysam


# ---------------------------------------------------------------------------
# MM tag parsing
# ---------------------------------------------------------------------------

MM_HEADER_RE = re.compile(r'^([ACGTUN])([+-])([A-Za-z0-9]+?)([?.]?)$')


def parse_mm_blocks(mm_str):
    """
    Parse an MM tag string into its constituent blocks.

    The MM tag has the form:  <base><strand><code>[?|.],n1,n2,...;<...>;
    e.g.:
        C+m?,5,12,0;C+h?,3,8;
        C+mh,5,12;            (combined codes, ML has 2 entries per position)

    Returns
    -------
    list of dict, one per block, each with:
        base, strand, mod_code, flag, n_positions
    Or {'raw_header': ..., 'parse_failed': True, 'n_positions': ...}
    if the header didn't match the expected shape.
    """
    blocks = []
    if not mm_str:
        return blocks
    for block in mm_str.rstrip(';').split(';'):
        if not block:
            continue
        parts = block.split(',')
        header = parts[0]
        positions = parts[1:]
        m = MM_HEADER_RE.match(header)
        if not m:
            blocks.append({'raw_header': header, 'parse_failed': True,
                           'n_positions': len(positions)})
            continue
        flag = m.group(4) or '(implicit)'
        blocks.append({
            'base': m.group(1),
            'strand': m.group(2),
            'mod_code': m.group(3),
            'flag': flag,
            'n_positions': len(positions),
        })
    return blocks


# ---------------------------------------------------------------------------
# Per-BAM survey
# ---------------------------------------------------------------------------

def survey_bam(bam_path, n_reads=1000):
    info = {'path': bam_path, 'basename': os.path.basename(bam_path)}

    bam = pysam.AlignmentFile(bam_path, 'rb')
    header = bam.header.to_dict()

    # Reference sequences
    sq_entries = header.get('SQ', [])
    info['n_references'] = len(sq_entries)
    info['reference_names_sample'] = [s['SN'] for s in sq_entries[:5]]
    info['reference_lengths_sample'] = [s.get('LN') for s in sq_entries[:5]]

    # Programs (basecaller, mappers, etc.)
    info['pg_entries'] = []
    for pg in header.get('PG', []):
        info['pg_entries'].append({
            'ID': pg.get('ID'),
            'PN': pg.get('PN'),
            'VN': pg.get('VN'),
            'CL': (pg.get('CL', '') or '')[:300],
        })

    # Read groups
    info['rg_entries'] = header.get('RG', [])

    # Sample reads
    mod_code_positions = Counter()                 # (base, strand, code) -> total positions
    flag_char_per_code = defaultdict(Counter)      # (base, strand, code) -> {flag: count}
    ml_values_aggregate = []                       # all ML bytes seen
    read_lengths = []
    mapqs = []
    strand_counter = Counter()
    has_mm_tag = 0
    has_ml_tag = 0
    n_examined = 0
    n_with_any_mod = 0
    parse_failures = 0

    for read in bam.fetch(until_eof=True):
        if n_examined >= n_reads:
            break
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        n_examined += 1
        read_lengths.append(read.query_length or 0)
        mapqs.append(read.mapping_quality)
        strand_counter['-' if read.is_reverse else '+'] += 1

        # MM tag (handle uppercase and legacy lowercase)
        mm = None
        for tag_name in ('MM', 'Mm'):
            try:
                mm = read.get_tag(tag_name)
                has_mm_tag += 1
                break
            except KeyError:
                continue

        # ML tag
        ml = None
        for tag_name in ('ML', 'Ml'):
            try:
                ml = read.get_tag(tag_name)
                has_ml_tag += 1
                break
            except KeyError:
                continue

        if mm:
            blocks = parse_mm_blocks(mm)
            had_any = False
            for blk in blocks:
                if blk.get('parse_failed'):
                    parse_failures += 1
                    continue
                key = (blk['base'], blk['strand'], blk['mod_code'])
                mod_code_positions[key] += blk['n_positions']
                flag_char_per_code[key][blk['flag']] += 1
                if blk['n_positions'] > 0:
                    had_any = True
            if had_any:
                n_with_any_mod += 1

        if ml is not None:
            ml_values_aggregate.extend(list(ml))

    bam.close()

    info['n_reads_examined'] = n_examined
    info['n_reads_with_any_mod'] = n_with_any_mod
    info['has_mm_tag_count'] = has_mm_tag
    info['has_ml_tag_count'] = has_ml_tag
    info['mm_block_parse_failures'] = parse_failures
    info['strand_distribution'] = dict(strand_counter)

    if read_lengths:
        info['read_length_stats'] = {
            'min': int(min(read_lengths)),
            'p25': int(np.percentile(read_lengths, 25)),
            'median': int(np.median(read_lengths)),
            'p75': int(np.percentile(read_lengths, 75)),
            'p95': int(np.percentile(read_lengths, 95)),
            'max': int(max(read_lengths)),
            'mean': int(np.mean(read_lengths)),
        }
    if mapqs:
        info['mapq_stats'] = {
            'min': int(min(mapqs)),
            'median': int(np.median(mapqs)),
            'max': int(max(mapqs)),
            'fraction_at_60': float((np.array(mapqs) == 60).mean()),
        }

    # Per-mod-code summary
    info['modifications'] = []
    for (base, strand, code), n_positions in sorted(mod_code_positions.items()):
        info['modifications'].append({
            'base': base,
            'strand': strand,
            'mod_code': code,
            'n_positions_total': int(n_positions),
            'flag_chars_observed': dict(flag_char_per_code[(base, strand, code)]),
        })

    # ML histogram (aggregate; per-mod-code attribution requires more parsing,
    # which we defer until we know whether multi-mod blocks are present)
    if ml_values_aggregate:
        ml_arr = np.array(ml_values_aggregate, dtype=np.int32)
        # ML byte N corresponds to probability bin [N/256, (N+1)/256].
        # For threshold P=0.2: byte values 0..51 (51/256 = 0.1992)
        # For threshold P=0.8: byte values 204..255 (204/256 = 0.7969)
        info['ml_histogram'] = {
            'n_total': int(len(ml_arr)),
            'mean_byte': float(ml_arr.mean()),
            'mean_prob': float(ml_arr.mean()) / 256.0,
            'fraction_p_below_0.2': float((ml_arr < 51).mean()),
            'fraction_p_in_0.2_to_0.8': float(((ml_arr >= 51) & (ml_arr < 204)).mean()),
            'fraction_p_at_or_above_0.8': float((ml_arr >= 204).mean()),
            'percentiles_byte': {
                'p1': int(np.percentile(ml_arr, 1)),
                'p10': int(np.percentile(ml_arr, 10)),
                'p25': int(np.percentile(ml_arr, 25)),
                'p50': int(np.percentile(ml_arr, 50)),
                'p75': int(np.percentile(ml_arr, 75)),
                'p90': int(np.percentile(ml_arr, 90)),
                'p99': int(np.percentile(ml_arr, 99)),
            },
            # 16 evenly spaced bins over byte range 0..256
            'histogram_16_bins': np.histogram(ml_arr, bins=16, range=(0, 256))[0].tolist(),
        }

    return info


# ---------------------------------------------------------------------------
# Example reads (raw MM/ML for eyeballing)
# ---------------------------------------------------------------------------

def show_example_reads(bam_path, n_examples=2, mm_truncate=300, seq_truncate=80):
    print(f"\n--- Example reads: {os.path.basename(bam_path)} ---")
    bam = pysam.AlignmentFile(bam_path, 'rb')
    n_shown = 0
    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if n_shown >= n_examples:
            break
        mm = None
        for tag_name in ('MM', 'Mm'):
            try:
                mm = read.get_tag(tag_name)
                break
            except KeyError:
                continue
        if mm is None:
            continue
        ml = None
        for tag_name in ('ML', 'Ml'):
            try:
                ml = read.get_tag(tag_name)
                break
            except KeyError:
                continue
        n_shown += 1
        seq = read.query_sequence or ''
        print(f"  Read #{n_shown}: name={read.query_name}, "
              f"len={len(seq)}, strand={'-' if read.is_reverse else '+'}, "
              f"chr={read.reference_name}, mapq={read.mapping_quality}")
        mm_show = mm if len(mm) <= mm_truncate else mm[:mm_truncate] + '...'
        print(f"    MM: {mm_show}")
        if ml is not None:
            ml_list = list(ml)
            print(f"    ML (n={len(ml_list)}): "
                  f"first 20 = {ml_list[:20]}{'...' if len(ml_list) > 20 else ''}")
        print(f"    seq[:{seq_truncate}]: {seq[:seq_truncate]}")
    bam.close()
    if n_shown == 0:
        print("  (no reads with MM tag found in initial scan)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_summary(info):
    print(f"\n=== {info.get('basename', info['path'])} ===")
    if 'error' in info:
        print(f"  ERROR: {info['error']}")
        return
    print(f"  References: {info['n_references']} total")
    print(f"    First few SN names: {info['reference_names_sample']}")
    print(f"  PG entries: {len(info['pg_entries'])}")
    for pg in info['pg_entries'][:6]:
        print(f"    - PN={pg['PN']!r}, VN={pg['VN']!r}, ID={pg['ID']!r}")
    print(f"  Reads examined: {info['n_reads_examined']}")
    print(f"    With MM tag: {info['has_mm_tag_count']}, "
          f"with ML tag: {info['has_ml_tag_count']}, "
          f"with any mod call: {info['n_reads_with_any_mod']}")
    print(f"  Strand: {info['strand_distribution']}")
    rl = info.get('read_length_stats', {})
    if rl:
        print(f"  Read length: min={rl['min']:,}, median={rl['median']:,}, "
              f"p95={rl['p95']:,}, max={rl['max']:,}")
    mq = info.get('mapq_stats', {})
    if mq:
        print(f"  MAPQ: min={mq['min']}, median={mq['median']}, "
              f"frac@60={mq['fraction_at_60']:.1%}")
    print(f"  Modifications observed:")
    for mod in info['modifications']:
        flags = ', '.join(f"{k!r}: {v}"
                          for k, v in mod['flag_chars_observed'].items())
        print(f"    {mod['base']}{mod['strand']}{mod['mod_code']}: "
              f"{mod['n_positions_total']:,} positions; flags = {{{flags}}}")
    h = info.get('ml_histogram')
    if h:
        print(f"  ML probability distribution ({h['n_total']:,} values, "
              f"mean P={h['mean_prob']:.3f}):")
        print(f"    P < 0.2: {h['fraction_p_below_0.2']:.1%}")
        print(f"    0.2 <= P < 0.8: {h['fraction_p_in_0.2_to_0.8']:.1%}  "
              f"<-- this is your discard rate at the 0.8/0.2 thresholds")
        print(f"    P >= 0.8: {h['fraction_p_at_or_above_0.8']:.1%}")
        print(f"    Byte percentiles: p10={h['percentiles_byte']['p10']}, "
              f"p50={h['percentiles_byte']['p50']}, "
              f"p90={h['percentiles_byte']['p90']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bams', nargs='+', help='BAM files to survey')
    parser.add_argument('--n-reads', type=int, default=1000,
                        help='Primary mapped reads to examine per BAM (default: 1000)')
    parser.add_argument('--out', type=str, default='bam_survey.json',
                        help='Path for JSON output (default: bam_survey.json)')
    parser.add_argument('--no-examples', action='store_true',
                        help='Skip the per-BAM example reads dump')
    parser.add_argument('--n-examples', type=int, default=2,
                        help='Number of example reads to print per BAM (default: 2)')
    args = parser.parse_args()

    results = []
    for bam_path in args.bams:
        if not os.path.exists(bam_path):
            print(f"[skip] not found: {bam_path}", file=sys.stderr)
            results.append({'path': bam_path, 'error': 'file not found'})
            continue
        print(f"[survey] {bam_path}", flush=True)
        try:
            info = survey_bam(bam_path, n_reads=args.n_reads)
            results.append(info)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            results.append({'path': bam_path, 'error': f'{type(e).__name__}: {e}'})

    # Save JSON
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[saved] {args.out}", flush=True)

    # Human-readable summary
    print("\n" + "=" * 78)
    print(" PER-BAM SUMMARY")
    print("=" * 78)
    for info in results:
        print_summary(info)

    if not args.no_examples:
        print("\n" + "=" * 78)
        print(" EXAMPLE READS (raw MM/ML)")
        print("=" * 78)
        for info in results:
            if 'error' in info:
                continue
            try:
                show_example_reads(info['path'], n_examples=args.n_examples)
            except Exception as e:
                print(f"  example-reads failed for {info['path']}: {e}")


if __name__ == '__main__':
    main()
