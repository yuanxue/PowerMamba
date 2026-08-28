#!/usr/bin/env bash
# One-shot reproduction of the two PowerMamba runs reported upstream.
#
#   bash run_repro.sh --bootstrap  # build .venv with the pinned stack (run once)
#   bash run_repro.sh              # preflight + data + both runs + compare
#   bash run_repro.sh --no-pred    # the 9-minute run only
#   bash run_repro.sh --check      # preflight only, no training
#
# Expects a Linux host with an NVIDIA GPU and Python 3.10. See SETUP.md.
#
# The pinned mamba_ssm 1.2.0.post1 publishes kernels for torch 1.12 through 2.3
# only. A host torch outside that range, such as the 2.7.0 that Lambda Stack
# ships, cannot load them and the import dies on a missing libcudart.so.11.0.
# --bootstrap builds an isolated venv holding torch 2.1.1+cu118, which is what
# the wheel tag expects.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CCV=v1.2.0.post1
MSV=v1.2.0.post1
WHEEL_TAG="cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64"
VENV="${ROOT}/.venv"
TRANSFORMERS_PIN="4.38.2"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\033[31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

CCW="https://github.com/Dao-AILab/causal-conv1d/releases/download/${CCV}/causal_conv1d-${CCV#v}+${WHEEL_TAG}.whl"
MSW="https://github.com/state-spaces/mamba/releases/download/${MSV}/mamba_ssm-${MSV#v}+${WHEEL_TAG}.whl"

# --------------------------------------------------------------- bootstrap
if [[ "${1:-}" == "--bootstrap" ]]; then
  say "Building ${VENV} with the pinned stack"
  command -v nvidia-smi >/dev/null 2>&1 || die "no nvidia-smi; this needs an NVIDIA GPU"

  # a mismatched wheel installed into the user site would shadow nothing inside
  # the venv, but clear it anyway so the host python is left clean
  pip uninstall -y mamba_ssm causal_conv1d >/dev/null 2>&1 || true

  python3 -m venv "$VENV" 2>/dev/null \
    || die "venv creation failed; run: sudo apt-get install -y python3.10-venv"
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  pip install -q -U pip wheel

  echo "installing torch 2.1.1+cu118 ..."
  pip install -q torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cu118
  pip install -q "numpy<2" pandas matplotlib scikit-learn

  echo "installing causal_conv1d + mamba_ssm ..."
  pip install -q "$CCW"
  pip install -q "$MSW"

  # mamba_ssm's __init__ imports a language-model head that pulls in
  # transformers, and transformers 5.x removed GreedySearchDecoderOnlyOutput.
  # Pin the last release that still exports it and imposes no torch floor.
  echo "pinning transformers ${TRANSFORMERS_PIN} ..."
  pip install -q "transformers==${TRANSFORMERS_PIN}"

  python -c "import torch; from mamba_ssm import Mamba; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| Mamba ok')"
  echo
  echo "Bootstrap done. Now run:  bash run_repro.sh"
  exit 0
fi

# use the pinned venv when it exists
if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

# ---------------------------------------------------------------- preflight
say "Preflight"

command -v nvidia-smi >/dev/null 2>&1 || die "no nvidia-smi; this needs an NVIDIA GPU (see SETUP.md)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

PYV="$(python -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "python ${PYV}"
[[ "$PYV" == "3.10" ]] || echo "  warning: the pinned wheels are cp310; ${PYV} needs a different wheel or a source build"

python - <<'PYCHK' || die "torch missing or not CUDA-enabled; run: bash run_repro.sh --bootstrap"
import sys, torch
print(f"torch {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
sys.exit(0 if torch.cuda.is_available() else 1)
PYCHK

# mamba_ssm 1.2.0.post1 ships kernels built per torch minor version, newest 2.3.
python - <<'PYCHK' || die "host torch cannot load the pinned mamba_ssm; run: bash run_repro.sh --bootstrap"
import sys, torch
mj, mn = (int(x) for x in torch.__version__.split('.')[:2])
cu = torch.version.cuda or ""
ok = (mj, mn) <= (2, 3)
if not ok:
    print(f"  torch {torch.__version__} is newer than the newest mamba_ssm 1.2.0.post1 build (torch 2.3)")
elif not cu.startswith("11.8"):
    print(f"  note: torch reports CUDA {cu} while the wheel tag is cu118")
sys.exit(0 if ok else 1)
PYCHK

python -c "from mamba_ssm import Mamba; print('mamba_ssm ok')" \
  || die "mamba_ssm import failed; run: bash run_repro.sh --bootstrap"

python - <<'PYCHK' || true
import transformers
mj = int(transformers.__version__.split('.')[0])
if mj >= 5:
    print(f"  warning: transformers {transformers.__version__} removed GreedySearchDecoderOnlyOutput;"
          f" run: pip install 'transformers==4.38.2'")
PYCHK

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

# The upstream scripts redirect all python output into logs/LongForecasting,
# so the terminal would otherwise sit silent for 25 minutes. Launch the run in
# the background and stream its log, which also makes a stall visible.
for r in "${RUNS[@]}"; do
  say "Training: ${r}  (50 epochs)"
  start=$(date +%s)

  marker="$(mktemp)"
  sh "./scripts/PowerMamba_${r}.sh" &
  trainpid=$!

  newlog=""
  for _ in $(seq 1 120); do
    newlog="$(find logs/LongForecasting -name '*.log' -newer "$marker" 2>/dev/null | head -1)"
    [[ -n "$newlog" ]] && break
    kill -0 "$trainpid" 2>/dev/null || break
    sleep 1
  done
  rm -f "$marker"

  tailpid=""
  if [[ -n "$newlog" ]]; then
    echo "streaming ${newlog}"
    tail -f --pid="$trainpid" "$newlog" &
    tailpid=$!
  else
    echo "no log appeared; the run may have failed at startup"
  fi

  wait "$trainpid" || die "training ${r} exited non-zero; see logs/LongForecasting"
  [[ -n "$tailpid" ]] && { kill "$tailpid" 2>/dev/null || true; wait "$tailpid" 2>/dev/null || true; }

  echo "elapsed: $(( ($(date +%s) - start) / 60 )) min"
done

# ------------------------------------------------------------------ compare
say "Results vs upstream logs"

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
    python3 - "$r" "$mse" "$tmse" "$mae" "$tmae" <<'PYCMP'
import sys
r = sys.argv[1]
mse, tmse, mae, tmae = map(float, sys.argv[2:])
d1 = (mse - tmse) / tmse * 100
d2 = (mae - tmae) / tmae * 100
print(f"{r:<12} {mse:<12.5f} {tmse:<12.5f} {d1:>+8.2f}%  {mae:<12.5f} {tmae:<12.5f} {d2:>+8.2f}%")
PYCMP
  done
done

cat <<'NOTE'

Metrics are in training-set standard deviations, not MW or $/MWh (exp_main.py
does not invert the scaler). GPU nondeterminism and a different card will move
these by a fraction of a percent; a delta of a few percent is a match, a delta
of tens of percent means something is wrong.
NOTE
