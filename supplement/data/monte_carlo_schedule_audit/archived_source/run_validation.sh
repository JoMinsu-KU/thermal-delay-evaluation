#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash run_validation.sh /absolute/project/root [--replicates 16] [--output NEW_DIR] [--resume] [--phase all|baseline|monte_carlo]' >&2
  exit 2
fi
validation_root="$(cd -- "$1" && pwd -P)"
shift
validation_here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "$validation_root"
source "$validation_root/scripts/activate_jev.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$validation_root/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
unset PYTHONPATH
exec python -B "$validation_here/run_suite.py" --root "$validation_root" "$@"
