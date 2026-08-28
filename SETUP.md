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

Install Python 3.10, since the pinned wheels are built for cp310.

```bash
conda env create -f environment.yml
conda activate PowerMamba

W=https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post1
pip install $W/causal_conv1d-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

W=https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1
pip install $W/mamba_ssm-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

The prebuilt wheels skip a source build of the CUDA kernels, which takes 20 to 40
minutes. A `cu122torch2.1` variant of each is published for CUDA 12 hosts.

## Running

`run_repro.sh` checks the environment, fetches the data, runs both trainings, and prints
the results next to upstream's committed numbers.

```bash
bash run_repro.sh            # preflight, data, both runs, comparison table
bash run_repro.sh --no-pred  # the 9-minute run only
bash run_repro.sh --check    # preflight only, no training
```

Run `--check` first on a fresh box. It verifies the GPU, the Python version, a
CUDA-enabled torch, and installs the two prebuilt wheels if `mamba_ssm` is missing.

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
