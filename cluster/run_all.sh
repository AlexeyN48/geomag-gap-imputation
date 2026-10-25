#!/bin/bash
# Весь конвейер по порядку на одном узле с 8 видеокартами (без планировщика).
# Каждый этап генерируется после окончания предыдущего: grid2 берёт лучшую
# скорость из grid1, seeds — победителей из select. Прерванный запуск
# продолжается той же командой: готовые задания пропускаются.
#   bash cluster/run_all.sh                 # все этапы
#   bash cluster/run_all.sh seeds summary   # только перечисленные
set -euo pipefail
cd "$(dirname "$0")/.."
export IGF_SPLIT=comp
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(baselines grid1 grid2 griddumps select seeds seeddumps stations stationdumps summary)
for st in "${STAGES[@]}"; do
  echo "=== $st"
  python -u scripts/bench_jobs.py "$st"
  python -u cluster/run_queue.py "cluster/jobs/$st.jsonl"
done
