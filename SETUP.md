# Setup and reproduction notes

A fork of [alimenati/PowerMamba](https://github.com/alimenati/PowerMamba) with a data
fetch script, a corrected training script, and the notes below. `git remote` keeps
`upstream` pointed at the original.

## Data

GridSet is published on Zenodo at [10.5281/zenodo.14451473](https://doi.org/10.5281/zenodo.14451473)
under CC-BY-4.0. The record is public, and the access token in the upstream Readme link
is not needed to download it.

```bash
bash data/fetch_gridset.sh                   # both files, 109 MB
bash data/fetch_gridset.sh --no-pred-only    # 8.6 MB file only
```

The script checks md5 against the Zenodo record and skips files already present. Both
CSVs are gitignored.

`GridSet_no_pred.csv` holds 43,824 hourly rows covering 2019-01-01 00:00 through
2023-12-31 23:00. Its 22 channels are 8 zonal loads (COAST, EAST, FWEST, NORTH, NCENT,
SOUTH, SCENT, WEST), 4 ancillary service prices (REGDN, REGUP, RRS, NSPIN), system-wide
wind and solar generation, and 8 settlement-point prices (LZ_AEN, LZ_CPS, LZ_HOUSTON,
LZ_LCRA, LZ_NORTH, LZ_RAYBN, LZ_SOUTH, LZ_WEST). The series has no gaps and no missing
values.

`GridSet_with_pred.csv` adds ERCOT's published 1h through 24h ahead forecasts for the 8
zonal loads, wind, and solar, reaching 262 channels. The forecast columns come first and
the 22 actual channels last.

## Environment

The model needs a Linux host with an NVIDIA GPU, because `mamba_ssm` ships compiled CUDA
kernels. Every file in `PowerMamba/models/` except `Transformer.py` imports `mamba_ssm`
at the top level, so CPU and Apple MPS are closed off even for the linear baselines.

Install Python 3.10, since the pinned wheels are built for cp310. Ubuntu 22.04 ships it
as the system Python.

The torch version matters as much as the CUDA version. `mamba_ssm` 1.2.0.post1 publishes
kernels compiled against torch 1.12 through 2.3, one build per minor version, and nothing
newer. A host carrying a later torch cannot load them: the import fails on a missing
`libcudart.so.11.0`. Lambda Stack ships torch 2.7.0 with CUDA 12.8 and hits exactly this,
so build an isolated environment rather than installing into the system Python.

```bash
bash run_repro.sh --bootstrap
```

That creates `.venv`, installs torch 2.1.1+cu118 with its supporting packages, adds the
two prebuilt kernel wheels, and verifies the import. `run_repro.sh` activates `.venv`
automatically on later calls. The same steps by hand:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" pandas matplotlib scikit-learn "transformers==4.38.2"

W=https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post1
pip install $W/causal_conv1d-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
W=https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1
pip install $W/mamba_ssm-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

`mamba_ssm`'s package init imports a language-model head, which imports `transformers`,
and `transformers` 5.x removed `GreedySearchDecoderOnlyOutput`. Version 4.38.2 is the
pin: it still exports the class, and it declares torch only as an optional extra, so it
does not fight the torch 2.1.1 pin. Any 4.x release through 4.57 still carries the class,
but the later ones want torch 2.2 or newer.

A driver new enough for CUDA 12 runs these cu118 binaries, so the host CUDA version is
not the constraint. The prebuilt wheels also skip a source build of the kernels, which
takes 20 to 40 minutes. `conda env create -f environment.yml` reaches the same place if
conda is already installed.

## Renting a GPU

The whole job is about 40 minutes of wall clock: roughly 10 minutes to build the conda
environment, a minute to fetch the data, and 25 minutes of training. Budget two hours of
billed time for a first attempt.

Ubuntu 22.04 ships Python 3.10, which matches the cp310 wheels, so a plain venv works and
conda is optional on either provider below.

### Lambda Labs

A10 24GB runs $1.29/GPU-hour and A6000 48GB runs $1.09, so the reproduction costs about
$2.20 to $2.60. The A6000 is the better pick on price and memory alike. Sign up, add an
SSH key to the workspace before launching, start a `gpu_1x_a10` or `gpu_1x_a6000`
instance, and connect as `ubuntu@<ip>`. Lambda Stack preinstalls the NVIDIA driver, CUDA,
and PyTorch on Ubuntu 22.04, and there is no quota approval to wait on. Capacity is the
thing that bites: instances are often unavailable in the popular regions, so check a few.

The V100 16GB at $0.79 is cheaper still, but its sm_70 architecture is older than what
the mamba_ssm kernels are routinely tested against, so it is a gamble rather than a
saving.

### AWS g5.xlarge

The instance carries an A10G 24GB with 4 vCPU and 16 GiB RAM at $1.006/hour on-demand in
us-east-1, putting the reproduction near $2.00 including a 100 GB gp3 root volume.

A new AWS account starts with a quota of zero vCPUs for "Running On-Demand G and VT
instances", and g5.xlarge needs four. Request the increase in Service Quotas before
anything else, because approval takes hours and sometimes a couple of business days.
After that, launch the Deep Learning AMI for Ubuntu 22.04, which carries the driver,
CUDA, and conda, give it a 100 GB root volume, open port 22 to your address, and ssh in.

Terminate the instance when the run finishes rather than stopping it. A stopped instance
still bills its EBS volume, and an instance left running costs $24 a day.

### Either one

```bash
git clone https://github.com/yuanxue/PowerMamba.git
cd PowerMamba && git checkout setup/gridshock-repro
bash run_repro.sh --bootstrap  # pinned torch 2.1.1+cu118 venv
bash run_repro.sh              # data + both runs + comparison
```

## Running

`run_repro.sh` checks the environment, fetches the data, runs both trainings, and prints
the results next to upstream's committed numbers.

```bash
bash run_repro.sh --bootstrap  # build .venv with the pinned stack, once per box
bash run_repro.sh              # preflight, data, both runs, comparison table
bash run_repro.sh --no-pred    # the 9-minute run only
bash run_repro.sh --check      # preflight only, no training
```

Run `--bootstrap` first on a fresh box, then `--check`. The check verifies the GPU, the
Python version, a CUDA-enabled torch, and that the host torch is old enough to load the
pinned kernels.

The underlying scripts still run directly:

```bash
cd PowerMamba
sh ./scripts/PowerMamba_no_pred.sh
sh ./scripts/PowerMamba_with_pred.sh
```

Upstream's committed logs put a 50-epoch run at 11.1 s per epoch without external
forecasts and 17.6 s with them. Both runs finish inside 30 minutes on a single GPU, so
an RTX 4090 rented by the hour covers the reproduction for well under a dollar.

## What the reported numbers mean

`PowerMamba/data_provider/data_loader.py` splits the series chronologically at 70/10/20
and fits a `StandardScaler` on the training block alone. The boundaries land here:

| split | rows | window |
|---|---|---|
| train | 30,676 | 2019-01-01 00:00 to 2022-07-02 03:00 |
| val   | 4,384  | 2022-07-02 04:00 to 2022-12-31 19:00 |
| test  | 8,764  | 2022-12-31 20:00 to 2023-12-31 23:00 |

Winter Storm Uri (February 2021) falls in the training block and Winter Storm Elliott
(December 2022) straddles the train and validation boundary. The test window is calendar
2023, which contains no comparable winter event.

`PowerMamba/exp/exp_main.py` calls `metric(preds, trues)` on scaled arrays without
inverting the scaler. Reported MSE and MAE are therefore in units of training-set
standard deviations, not MW or $/MWh, and rescaling comes first when comparing against
results reported in physical units.

Upstream's committed logs end at these values, which are the reproduction target:

| run | MSE | MAE | RSE |
|---|---|---|---|
| no external forecasts | 0.12964 | 0.16612 | 0.28324 |
| with external forecasts | 0.07367 | 0.12369 | 0.21351 |

## Changes from upstream

`PowerMamba/scripts/PowerMamba_no_pred.sh` passed `--train_epochs 1` while its own
committed log records 50 epochs, so the released script reproduced nothing. This fork
sets 50. The fix is worth sending upstream.

Both scripts write their log to `logs/LongForecasting/` while creating only `./logs`,
so the redirect failed and Python never started. Both now create the nested directory.

Baseline run configurations are missing upstream. `PowerMamba/models/` carries
Autoformer, iTransformer, PatchTST, TimesNet, DLinear, TimeMachine, and Transformer,
while `PowerMamba/scripts/` carries only the two PowerMamba runs. Rebuilding the
paper's comparison tables means recovering baseline hyperparameters first.

Upstream publishes no license file, which leaves the code all rights reserved. The
Zenodo data is CC-BY-4.0.
