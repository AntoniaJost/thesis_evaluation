#!/bin/bash
#SBATCH --job-name=bias_maps
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --account=bk1450
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err
source ~/.bashrc
conda activate thesis_eval
set -euo pipefail

# if running for the first time and bias maps are wanted, calculate the corresponding csv files first! only needed once
# python -m evaluation.range_summary

python -m evaluation.main run_plots='["bias_map"]' out.overwrite=true plots.bias_map.coastline_colour=black plots.bias_map.models=["free_run_control", "forced_sst", "forced_sst_2k", "forced_sst_4k"]

python -m evaluation.main run_plots='["bias_map"]' out.overwrite=true plots.bias_map.coastline_colour=black plots.bias_map.models=["free_run_prediction"] plots.bias_map.time.use_named=TSTP

echo "ALL DONE."