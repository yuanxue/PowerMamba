# Reproduction results

Measured on a Lambda Cloud A10 24GB, driver 580.105.08, with the pinned stack from
`run_repro.sh --bootstrap`: Python 3.10, torch 2.1.1+cu118, mamba_ssm 1.2.0.post1,
transformers 4.38.2. Upstream values are the final lines of the logs committed at
`PowerMamba/logs/`.

Metrics are in training-set standard deviations. `exp/exp_main.py` scores the scaled
arrays without inverting the `StandardScaler`, so none of these numbers are MW or $/MWh.

## With external forecasts

The released script and its committed log agree on every argument, and the run
reproduces.

| metric | measured | upstream | delta |
|---|---|---|---|
| MSE | 0.075066 | 0.073669 | +1.90% |
| MAE | 0.124825 | 0.123690 | +0.92% |
| RSE | 0.215527 | 0.213512 | +0.94% |

A residual near 1 to 2% is what a different card and nondeterministic reduction order
produce. Per-category test losses at epoch 50: ancillary services 0.0188, price 0.0931,
wind 0.0936, solar 0.2037, load 0.0675.

## Without external forecasts

The released script passed `--dropout 0.2` while its committed log records `dropout=0.7`.
Training at 0.2 gives:

| metric | measured | upstream | delta |
|---|---|---|---|
| MSE | 0.134360 | 0.129643 | +3.64% |
| MAE | 0.171710 | 0.166120 | +3.36% |
| RSE | 0.288347 | 0.283241 | +1.80% |

The gap is twice what the with-forecast run shows on the same card, which points at the
dropout rather than at hardware. The script now sets 0.7; a rerun measuring that is the
open item.

## Cost

The two runs took about 9 and 15 minutes at 11 and 18 seconds per epoch, which is roughly
$0.55 of A10 time at $1.29 per hour. Bootstrapping the environment and fetching 109 MB of
data adds another 10 minutes.

## Defects found in the upstream release

1. `PowerMamba_no_pred.sh` passed `--train_epochs 1` against a committed log recording 50.
2. Both scripts wrote their log into `logs/LongForecasting/` while creating only `./logs`,
   so the redirect failed and Python never started.
3. `PowerMamba_no_pred.sh` passed `--dropout 0.2` against a committed log recording 0.7.
4. Baseline run configurations are absent, so the paper's comparison tables cannot be
   rebuilt without recovering hyperparameters for the seven baseline models.

The environment is a separate obstacle. `mamba_ssm` 1.2.0.post1 publishes kernels for
torch 1.12 through 2.3, and `transformers` 5.0.0 removed a class its package init imports,
so a 2026 host installing the pinned requirements fails twice before reaching the data.
