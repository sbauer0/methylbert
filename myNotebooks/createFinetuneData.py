from functools import partial
from methylbert.data.finetune_data_generate import finetune_data_generate
from methylbert.data.nanopore.finetune_extract import ont_read_extract

# sc_dataset file lists BOTH run-1 BAMs with their cell-type labels, tab-sep:
#   /tmp/bauerste/colo829/PAU59949....bam      T
#   /tmp/bauerste/colo829bl/PAU59807....bam    N

finetune_data_generate(
    f_dmr="dmrs.tsv",
    output_dir="/tmp/bauerste/finetune_run1",
    f_ref="/home/bauerste/GRCh38_no_alt_analysis_set/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna",
    sc_dataset="run1_bams.tsv",
    n_mers=3,
    n_dmrs=50,                       # top-50 per ctype -> 100 DMRs, selected here
    split_ratio=0.85,                # 85% train / 15% eval, split BY READ NAME
    ignore_sex_chromo=False,         # see caution 2 (keeps chrX)
    methyl_caller="dorado",
    read_extract_sequences_func=partial(ont_read_extract, mm_flag="?"),
)

print('ok')