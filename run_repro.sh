#!/usr/bin/env bash
# One-shot reproduction of the two PowerMamba runs reported upstream.
#
#   bash run_repro.sh              # preflight + data + both runs + compare
#   bash run_repro.sh --no-pred    # the 9-minute run only
#   bash run_repro.sh --check      # preflight only, no training
#
# Expects a Linux host with an NVIDIA GPU and Python 3.10. See SETUP.md.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CCV=v1.2.0.post1
MSV=v1.2.0.post1
WHEEL_TAG="cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\033[31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
say "Preflight"

command -v nvidia-smi >/dev/null 2>&1 || die "no nvidia-smi; this needs an NVIDIA GPU (see SETUP.md)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

PYV="$(python -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "python ${PYV}"
[[ "$PYV" == "3.10" ]] || echo "  warning: pinned wheels are cp310; ${PYV} will need a different wheel or a source build"

python - <<'PY' || die "torch missing or not CUDA-enabled; run: conda env create -f environment.yml"
import torch
print(f"torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
assert torch.cuda.is_available()
PY

if ! python -c "import mamba_ssm" 2>/dev/null; then
  say "Installing mamba_ssm + causal_conv1d (prebuilt wheels)"
  pip install -q "https://github.com/Dao-AILab/causal-conv1d/releases/download/${CCV}/causal_conv1d-${CCV#v}+${WHEEL_TAG}.whl"
  pip install -q "https://github.com/state-spaces/mamba/releases/download/${MSV}/mamba_ssm-${MSV#v}+${WHEEL_TAG}.whl"
fi
python -c "import mamba_ssm; print('mamba_ssm ok')" || die "mamba_ssm import failed"

[[ "${1:-}" == "--check" ]] && { echo; echo "Preflight passed."; exit 0; }

# --------------------------------------------------------------------- data
say "Data"
if [[ "${1:-}" == "--no-pred" ]]; then
  bash "${ROOT}/data/fetch_gridset.sh" --no-pred-only
  RUNS=("no_pred")
else
  bash "${ROOT}/data/fetch_gridset.sh"
  RUNS=("no_pred" "with_pred")
fi

# ------------------------------------------------------------------ training
cd "${ROOT}/PowerMamba"
mkdir -p logs/LongForecasting

for r in "${RUNS[@]}"; do
  say "Training: ${r}  (50 epochs)"
  start=$(date +%s)
  sh "./scripts/PowerMamba_${r}.sh"
  echo "elapsed: $(( ($(date +%s) - start) / 60 )) min"
done

# ------------------------------------------------------------------ compare
say "Results vs upstream logs"

# run:target_mse:target_mae
TARGETS=("no_pred:0.12964342534542084:0.16612033545970917"
         "with_pred:0.07366887480020523:0.12368996441364288")

printf '%-12s %-12s %-12s %-10s %-12s %-12s %-10s\n' run mse mse_ref delta mae mae_ref delta
for r in "${RUNS[@]}"; do
  for t in "${TARGETS[@]}"; do
    IFS=':' read -r name tmse tmae <<< "$t"
    [[ "$name" == "$r" ]] || continue
    line="$(grep -h -oE 'mse:[0-9.]+, mae:[0-9.]+' logs/LongForecasting/*.log 2>/dev/null | tail -1 || true)"
    if [[ -z "$line" ]]; then
      printf '%-12s %s\n' "$r" "no result line found in logs/LongForecasting/*.log"
      continue
    fi
    mse="${line#mse:}"; mse="${mse%%,*}"
    mae="${line##*mae:}"
    python3 - "$r" "$mse" "$tmse" "$mae" "$tmae" <<'PY'
import sys
r, mse, tmse, mae, tmae = sys.argv[1], *map(float, sys.argv[2:])
d1 = (mse - tmse) / tmse * 100
d2 = (mae - tmae) / tmae * 100
print(f"{r:<12} {mse:<12.5f} {tmse:<12.5f} {d1:>+8.2f}%  {mae:<12.5f} {tmae:<12.5f} {d2:>+8.2f}%")
PY
  done
done

cat <<'NOTE'

Metrics are in training-set standard deviations, not MW or $/MWh (exp_main.py
does not invert the scaler). GPU nondeterminism and a different card will move
these by a fraction of a percent; a delta of a few percent is a match, a delta
of tens of percent means something is wrong.
NOTE
