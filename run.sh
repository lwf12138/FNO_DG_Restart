#!/usr/bin/env bash
set -euo pipefail
dataset_root="$(cd "$(dirname "$0")" && pwd)"
cd "$dataset_root"
if [[ -n "${DG_PYTHON:-}" ]]; then
  dataset_python="$DG_PYTHON"
elif [[ -x "$dataset_root/.venv/bin/python" ]]; then
  dataset_python="$dataset_root/.venv/bin/python"
else
  dataset_python="$(command -v python3)"
fi
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$dataset_root/cache/matplotlib"
mkdir -p logs checks cache data reference runs
"$dataset_python" -B prepare_solver.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > logs/build.log 2>&1
cmake --build build --parallel 2 >> logs/build.log 2>&1
"$dataset_python" -B generate_dataset.py "$@" 2>&1 | tee logs/generation.log
