#!/bin/bash
# Launch bench_multinode.py across multiple nodes.
#
# Run this on EACH node (or fan out via pdsh/srun/PyTorchJob).
# All nodes must run the same command with the same NNODES/NPROC.
#
# Usage:
#   ./run_multinode.sh [NNODES] [NPROC_PER_NODE] [MASTER_ADDR] [MASTER_PORT] [-- extra args]
#
# Examples:
#   ./run_multinode.sh 3 2 10.0.0.1 29500
#   ./run_multinode.sh 3 2 master-0.headless 29500 -- --section collectives
#   DTYPE=fp16 ./run_multinode.sh 3 2 10.0.0.1 29500
#
# Environment variables MASTER_ADDR and MASTER_PORT are used as defaults.
# DTYPE (default bf16) selects the run dtype and suffixes the result file,
# matching run_all.sh's <bench>_tp<N>_<dtype>.json naming; run once per
# dtype you want.

set -uo pipefail

NNODES="${1:-3}"
NPROC="${2:-2}"
MASTER="${3:-${MASTER_ADDR:-localhost}}"
PORT="${4:-${MASTER_PORT:-29500}}"
DTYPE="${DTYPE:-bf16}"

# Consume positional args; everything after "--" goes to bench_multinode.py
shift 4 2>/dev/null || true
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done
for arg in "${EXTRA_ARGS[@]}"; do
    case "$arg" in
        --dtype|--dtype=*) echo "error: set DTYPE=<bf16|fp16|fp32> instead of passing --dtype (it names the result file)" >&2; exit 1 ;;
        --json|--json=*)   echo "error: the result path is derived from NNODES/NPROC/DTYPE; do not pass --json" >&2; exit 1 ;;
    esac
done

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
JSON_DIR="${BENCH_DIR}/results"
mkdir -p "$JSON_DIR"

JSON_PATH="${JSON_DIR}/multinode_${NNODES}n${NPROC}g_${DTYPE}.json"

echo "============================================================"
echo "pytorch-dist-bench: multi-node benchmark"
echo "  Nodes: ${NNODES}  |  GPUs/node: ${NPROC}"
echo "  Master: ${MASTER}:${PORT}  |  dtype: ${DTYPE}"
echo "  Results: ${JSON_PATH}"
echo "============================================================"

torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER}:${PORT}" \
    "${BENCH_DIR}/bench_multinode.py" \
    --json "$JSON_PATH" \
    --dtype "$DTYPE" \
    "${EXTRA_ARGS[@]}"
