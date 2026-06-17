#!/usr/bin/env bash
set -euo pipefail
TH=${1:-0}
K=/tmp/kubectl
KC=/root/kubectl-tione.conf
NS=maas-public
P0=glm5-p800-flexkv-inference-prefill-0
P1=glm5-p800-flexkv-inference-prefill-0-1
DATA=/workspace/zittozhang/glm5-online-data-24K-64K_first20.json
BASE=http://127.0.0.1:30000/v1/
MODEL=glm-5
OUT=/root/zittozhang/min_layerwise_load_active_threshold_${TH}_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$OUT") 2>&1
echo "RESULT_LOG=$OUT"
echo "@@ENV threshold=$TH"
for pod in "$P0" "$P1"; do
  echo "== $pod =="
  $K --kubeconfig="$KC" exec -n "$NS" "$pod" -c prefill-engine -- bash -lc "pid=\$(pgrep -f sglang.launch_server | head -1); tr '\0' '\n' < /proc/\$pid/environ | grep -E 'FLEXKV_MIN_LAYERWISE_LOAD_TOKENS|FLEXKV_LAYERWISE_PERSISTENT_GPU_ISSUE|XSGL_TRANSFER_H2D_SEGMENT_THRESHOLD|XSGL_TRANSFER_D2H_SEGMENT_THRESHOLD' | sort"
done
run_bench() {
  local phase=$1
  echo "@@BENCH_START phase=$phase threshold=$TH"
  $K --kubeconfig="$KC" exec -n "$NS" "$P0" -c prefill-engine -- bash -lc "source /root/miniconda/etc/profile.d/conda.sh && conda activate flexkv_env && cd /workspace/zittozhang && python3 -u bounded_flexkv_benchmark.py --data $DATA --base-url $BASE --model $MODEL --concurrency 4 --limit 20 --max-tokens 1"
  echo "@@BENCH_END phase=$phase threshold=$TH"
}
collect_stats() {
  local phase=$1
  for pod in "$P0" "$P1"; do
    echo "@@STATS threshold=$TH phase=$phase pod=$pod"
    $K --kubeconfig="$KC" exec -i -n "$NS" "$pod" -c prefill-engine -- python3 - <<'PY'
import glob
import os
import json
p = sorted(glob.glob('/workspace/zittozhang/logs/standalone_*.log'), key=os.path.getmtime)[-1]
text = open(p, errors='ignore').read()
errors = [ln for ln in text.splitlines() if 'ERROR' in ln or 'Traceback' in ln or 'Exception' in ln]
print(json.dumps({
    'log': p,
    'start_store': text.count('start_store_kv'),
    'start_load': text.count('start_load_kv'),
    'D2H': text.count('D2H transfer request'),
    'H2D': text.count('H2D transfer request'),
    'LAYERWISE': text.count('LAYERWISE transfer request'),
    'skip_small_load': text.count('skip_small_load'),
    'STORE_TXN_queue': text.count('[STORE-TXN] queue'),
    'length_mismatch': text.count('length mismatch'),
    'real_error_lines': len(errors),
}, ensure_ascii=False))
PY
  done
}
echo "@@STORE threshold=$TH"
run_bench store
collect_stats store
echo "@@FLUSH threshold=$TH"
$K --kubeconfig="$KC" exec -n "$NS" "$P0" -c prefill-engine -- curl -s -X POST -w '\nHTTP_CODE=%{http_code}\n' http://localhost:30000/flush_cache
echo "@@REPEAT threshold=$TH"
run_bench repeat
collect_stats repeat
echo "ALL_DONE threshold=$TH RESULT_LOG=$OUT"
