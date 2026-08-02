#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ours self-view:
# - select the highest-scoring self case from the similarity Top-10 pool
# - if an opposite-outcome case exists among the remaining Top-10 cases,
#   append the highest-scoring such case as a counterexample
export DATASETS="${DATASETS:-XES}"
export RUNS="${RUNS:-1}"
export SELF_TOPK=1
export SELF_COUNTER=1

exec "${SCRIPT_DIR}/run_steer_kt.sh"
