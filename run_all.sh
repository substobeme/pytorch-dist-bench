#!/bin/bash
# Run all pytorch-dist-bench benchmarks sequentially on N GPUs.
#
# Sequential execution prevents GPU contention that corrupts measurements.
# Each benchmark writes JSON to the results directory.
#
# Each benchmark declares the dtypes it is swept over in a `DTYPES = (...)`
# line at the top of its script (with the reason next to it); it is run
# once per dtype, writing <bench>_tp<N>_<dtype>.json. Benchmarks without a
# --dtype option have no DTYPES line and write <bench>_tp<N>.json.
#
# Usage:
#   ./run_all.sh [nproc] [--json-dir DIR] [--dtypes "bf16 fp16"]
#
# Examples:
#   ./run_all.sh 8                          # 8 GPUs, results in ./results/
#   ./run_all.sh 2 --json-dir /tmp/bench    # 2 GPUs, results in /tmp/bench/
#   ./run_all.sh 8 --dtypes bf16            # only bf16 (subset of each sweep)

set -uo pipefail

NPROC="${1:-8}"
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
JSON_DIR="${BENCH_DIR}/results"
TIMEOUT="${BENCH_TIMEOUT:-600}"
DTYPES_OVERRIDE=""

shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --json-dir) JSON_DIR="$2"; shift 2 ;;
        --dtypes) DTYPES_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$JSON_DIR"

for d in $DTYPES_OVERRIDE; do
    case "$d" in
        bf16|fp16|fp32) ;;
        *) echo "error: --dtypes takes space-separated names from: bf16 fp16 fp32 (got '$d')" >&2
           exit 1 ;;
    esac
done

BENCHMARKS=(
    bench_verify
    bench_collectives
    bench_symm_mem_fused_ops
    bench_fp8_fused_ops
    bench_migration_path
    bench_inference_tp_layer
    bench_inference_tp_vllm
    bench_training_fsdp_collectives
    bench_fsdp2_training
    bench_pipeline_parallel
    bench_moe_alltoall
    bench_allreduce_dispatch
    bench_compile_distributed
    bench_e2e
)

PASSED=()
FAILED=()

echo "============================================================"
echo "pytorch-dist-bench: ${#BENCHMARKS[@]} benchmarks, one run per declared dtype"
echo "  GPUs: ${NPROC}"
echo "  Results: ${JSON_DIR}"
echo "  Timeout: ${TIMEOUT}s per run (override: BENCH_TIMEOUT=N)"
[[ -n "$DTYPES_OVERRIDE" ]] && echo "  Dtypes: restricted to ${DTYPES_OVERRIDE}"
echo "============================================================"

run_one() {
    local bench_name="$1" json_path="$2" dtype="${3:-}"
    local label="${bench_name}${dtype:+/$dtype}"
    echo ""
    echo "--- ${label} ---"
    if timeout "$TIMEOUT" torchrun --nproc_per_node="$NPROC" \
        "${BENCH_DIR}/${bench_name}.py" --json "$json_path" \
        ${dtype:+--dtype "$dtype"} 2>&1; then
        PASSED+=("$label")
    else
        FAILED+=("$label")
        echo "  FAILED: ${label}"
    fi
}

# The DTYPES tuple from a script's source, as space-separated names.
declared_dtypes() {
    sed -n 's/^DTYPES = (\(.*\))$/\1/p' "$1" | tr -d '",'
}

for bench_name in "${BENCHMARKS[@]}"; do
    script="${BENCH_DIR}/${bench_name}.py"
    if [[ ! -f "$script" ]]; then
        echo "error: $script not found" >&2
        exit 1
    fi
    dtypes=$(declared_dtypes "$script")
    if [[ -z "$dtypes" ]] && grep -q -- '"--dtype"' "$script"; then
        echo "error: ${bench_name}.py takes --dtype but declares no DTYPES" >&2
        exit 1
    fi
    if [[ -z "$dtypes" ]]; then
        run_one "$bench_name" "${JSON_DIR}/${bench_name}_tp${NPROC}.json"
        continue
    fi
    for dtype in $dtypes; do
        # --dtypes restricts the declared sweep; it never adds to it.
        if [[ -n "$DTYPES_OVERRIDE" && " $DTYPES_OVERRIDE " != *" $dtype "* ]]; then
            continue
        fi
        run_one "$bench_name" \
            "${JSON_DIR}/${bench_name}_tp${NPROC}_${dtype}.json" "$dtype"
    done
done

echo ""
echo "============================================================"
echo "Results: ${#PASSED[@]}/$(( ${#PASSED[@]} + ${#FAILED[@]} )) passed"
for b in "${PASSED[@]}"; do echo "  OK:   $b"; done
for b in "${FAILED[@]}"; do echo "  FAIL: $b"; done
echo "JSON: ${JSON_DIR}/"
echo "============================================================"

[ ${#FAILED[@]} -eq 0 ]
