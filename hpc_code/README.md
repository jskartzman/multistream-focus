# HPC Experiments

This directory contains scripts for running change point detection experiments on high-performance computing systems.

## Experiments

### ARL Experiments (run_arl_experiments.py)

Average Run Length experiments measure false alarm rates when there is NO change in the data. Higher ARL values indicate better performance (fewer false alarms).

#### Usage

```bash
python run_arl_experiments.py --algorithms <algorithms> [options]
```

#### Required Arguments

- `--algorithms`: Comma-separated list of algorithms to test
  - Available: `focus_decay`, `focus_nonuhat`, `xumei`
  - Example: `"focus_decay,xumei"`

#### Optional Arguments

- `--Ms`: Number of streams to test (default: "1,3,5,10")
- `--threshold-min`: Minimum threshold value (default: 1000)
- `--threshold-max`: Maximum threshold value (default: 5000)
- `--threshold-steps`: Number of threshold steps to test (default: 5)
- `--T`: Time horizon for each simulation (default: 1000000)
- `--sims`: Number of simulations per configuration (default: 50)
- `--workers`: Number of parallel workers (default: auto-detect)
- `--xumei-lb`: Lower bound parameter for xumei algorithm (default: 2.0)
- `--save`: Save results to pickle files

#### Example

```bash
python run_arl_experiments.py \
    --algorithms "focus_decay,xumei" \
    --Ms "5,10" \
    --threshold-min 10 \
    --threshold-max 100 \
    --threshold-steps 10 \
    --T 1000000 \
    --sims 500 \
    --save
```

### EDD Experiments (run_edd_experiments.py)

Expected Detection Delay experiments measure how quickly algorithms detect actual changes in the data. Lower EDD values indicate better performance (faster detection).

#### Usage

```bash
python run_edd_experiments.py --algorithms <algorithms> [options]
```

#### Required Arguments

- `--algorithms`: Comma-separated list of algorithms to test
  - Available: `focus_decay`, `focus_oracle`, `focus_nonuhat`, `xumei`
  - Example: `"focus_decay,xumei"`

#### Optional Arguments

- `--nus`: Change point locations to test (default: "0,1000,10000")
- `--Ms`: Number of streams to test (default: "10")
- `--threshold-min`: Minimum threshold value (default: 1000)
- `--threshold-max`: Maximum threshold value (default: 5000)
- `--threshold-steps`: Number of threshold steps to test (default: 5)
- `--T`: Time horizon for each simulation (default: 1000000)
- `--mu1`: Post-change mean value (default: 1.0)
- `--sims`: Number of simulations per configuration (default: 50)
- `--workers`: Number of parallel workers (default: auto-detect)
- `--xumei-lb`: Lower bound parameter for xumei algorithm (default: 0.1)
- `--save`: Save results to pickle files

#### Example

```bash
python run_edd_experiments.py \
    --algorithms "focus_decay,xumei" \
    --nus "0,1000,5000,10000" \
    --Ms "5,10,15,20" \
    --threshold-min 1000 \
    --threshold-max 10000 \
    --threshold-steps 10 \
    --T 1000000 \
    --mu1 1.0 \
    --sims 500 \
    --save
```

## Output

When using `--save`, results are saved as pickle files with timestamps in the filename. Files include mean and standard deviation results for each configuration tested.

## SLURM Batch Scripts

- `run_arl_slurm.sbatch`: Template for running ARL experiments on SLURM clusters
- `run_edd_slurm.sbatch`: Template for running EDD experiments on SLURM clusters

Modify the SLURM parameters and file paths as needed for your cluster environment. 