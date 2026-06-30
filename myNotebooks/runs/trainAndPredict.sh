#!/bin/bash
# afk_train.sh — run from /home/bauerste/methylbert_finetune/methylbert/myNotebooks/runs
# Trains then dumps. Each python script handles its own GPU isolation (subprocess-per-run)
# and skip-if-exists, so a re-run resumes. set -e is intentionally OFF so one bad stage
# doesn't abort the rest.
set -u

source ~/venvs/5basemethylbert/bin/activate

cd /home/bauerste/methylbert_finetune/methylbert/myNotebooks/runs

log() { echo "[$(date '+%F %H:%M:%S')] $*"; }

log "=== TRAINING ==="
log "dmr100 (3 seeds x 2 variants)"; python fullrun_finetune.py  > ft_dmr100.log  2>&1 || log "dmr100 nonzero exit"
log "dmr250 (seed42 x 2)";          python finetune_dmr250.py   > ft_dmr250.log  2>&1 || log "dmr250 nonzero exit"
log "dmr500 (seed42 x 2)";          python finetune_dmr500.py   > ft_dmr500.log  2>&1 || log "dmr500 nonzero exit"
log "dmr1000 (seed42 x 2)";         python finetune_dmr1000.py  > ft_dmr1000.log 2>&1 || log "dmr1000 nonzero exit"

log "=== DUMPING PREDICTIONS ==="
log "dump dmr100";  python dump_all_dmr100.py  > dump_dmr100.log  2>&1 || log "dump100 nonzero exit"
log "dump dmr250";  python dump_all_dmr250.py  > dump_dmr250.log  2>&1 || log "dump250 nonzero exit"
log "dump dmr500";  python dump_all_dmr500.py  > dump_dmr500.log  2>&1 || log "dump500 nonzero exit"
log "dump dmr1000"; python dump_all_dmr1000.py > dump_dmr1000.log 2>&1 || log "dump1000 nonzero exit"

log "=== ALL DONE ==="