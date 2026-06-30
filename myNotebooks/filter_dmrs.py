#!/usr/bin/env python3
"""Pre-filter dmrs_no_id_NMS.tsv before fine-tune data generation.
Drops (1) DMRs on unplaced/alt contigs (chr name contains '_') and
(2) mapping-pileup DMRs whose pooled run-1 mean per-base depth over the DMR
interval exceeds MAX_DEPTH. Uses one `samtools bedcov` pass per BAM (fast),
preserves descending-areaStat order and per-ctype balance, and writes a
filtered TSV plus a dropped-DMR log."""
import os, subprocess, sys, tempfile
import pandas as pd

HOME      = os.path.expanduser("~")
IN_TSV    = f"{HOME}/methylbertDMRs/dmrs_no_id_NMS.tsv"
OUT_TSV   = f"{HOME}/methylbertDMRs/dmrs_filtered.tsv"
LOG_TSV   = f"{HOME}/methylbertDMRs/dmrs_dropped.tsv"
MAX_DEPTH = 165          # 3x the ~55x chr20 baseline; mean per-base depth over the interval
MIN_PER_CLASS_KEEP = 500 # need >=500 of each ctype for the 1000-DMR set (500/class)
MIN_MAPQ  = 1            # bedcov -Q; raise to 20 to match -q 20 filtering exactly
BAMS = [
    "/tmp/bauerste/COLO829_BL/colo829/PAU59949.d052sup4305mCG_5hmCGvHg38_pass.bam",
    "/tmp/bauerste/COLO829_BL/colo829bl/PAU59807.d052sup4305mCG_5hmCGvHg38_pass.bam",
]

df = pd.read_csv(IN_TSV, sep="\t").reset_index(drop=True)
print(f"input DMRs: {len(df)} | ctype balance: {df['ctype'].value_counts().to_dict()}")

unplaced = df["chr"].str.contains("_")
print(f"unplaced/alt-contig DMRs: {int(unplaced.sum())}")
placed = df[~unplaced].copy()

# write one BED of all placed intervals (BED is 0-based half-open; keep df index in col 4)
with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as bed:
    for idx, r in placed.iterrows():
        bed.write(f"{r['chr']}\t{int(r['start'])-1}\t{int(r['end'])}\t{idx}\n")
    bed_path = bed.name

def bedcov(bam):
    r = subprocess.run(["samtools", "bedcov", "-Q", str(MIN_MAPQ), bed_path, bam],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"bedcov failed on {bam}\n{r.stderr}")
    out = {}
    for line in r.stdout.strip().splitlines():
        f = line.split("\t")
        out[int(f[3])] = int(f[4])          # idx -> summed per-base depth
    return out

print("counting coverage (one bedcov pass per BAM)...", flush=True)
depth = {}
for bam in BAMS:
    for idx, v in bedcov(bam).items():
        depth[idx] = depth.get(idx, 0) + v
os.unlink(bed_path)

placed["interval_len"] = placed["end"].astype(int) - (placed["start"].astype(int) - 1)
placed["mean_depth"]   = placed.index.map(lambda i: depth.get(i, 0)) / placed["interval_len"]
placed["drop"]         = placed["mean_depth"] > MAX_DEPTH

kept = placed[~placed["drop"]].sort_index()
dropped = pd.concat([
    placed[placed["drop"]].assign(reason=f"depth>{MAX_DEPTH}"),
    df[unplaced].assign(mean_depth=float("nan"), reason="unplaced_contig"),
]).sort_index()

kept[["chr", "start", "end", "ctype", "areaStat"]].to_csv(OUT_TSV, sep="\t", index=False)
dropped.to_csv(LOG_TSV, sep="\t", index=False)

print(f"\nkept: {len(kept)} | dropped: {len(dropped)} "
      f"({dropped['reason'].value_counts().to_dict()})")
print(f"kept ctype balance: {kept['ctype'].value_counts().to_dict()}")
for c, cnt in kept["ctype"].value_counts().items():
    warn = f"  <-- WARNING < {MIN_PER_CLASS_KEEP}" if cnt < MIN_PER_CLASS_KEEP else ""
    print(f"  ctype {c}: {cnt}{warn}")

print("\nkept-DMR mean-depth distribution (is the cap in the empty valley?):")
print(kept["mean_depth"].describe(percentiles=[.5, .9, .95, .99]).round(1).to_string())
if len(dropped):
    print("\ndropped DMRs (first 15, ranked):")
    print(dropped.head(15)[["chr", "start", "end", "ctype", "mean_depth", "reason"]]
          .to_string(index=False))
print(f"\nwrote {OUT_TSV}\nwrote {LOG_TSV}")