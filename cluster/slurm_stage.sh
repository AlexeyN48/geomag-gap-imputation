#!/bin/bash
# Один этап конвейера на узле с 8 видеокартами под SLURM.
#   sbatch cluster/slurm_stage.sh seeds
# Параметры #SBATCH (раздел, время, память) подправьте под свой кластер.
#SBATCH --job-name=geomag
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --output=cluster/logs/slurm_%x_%j.out
set -euo pipefail
STAGE="${1:?укажите этап: baselines grid1 grid2 griddumps select seeds seedval finalists stations stationval test summary}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export IGF_SPLIT=comp
python -u scripts/bench_jobs.py "$STAGE"
python -u cluster/run_queue.py "cluster/jobs/$STAGE.jsonl"
