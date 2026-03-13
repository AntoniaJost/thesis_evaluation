#!/bin/bash
#SBATCH --job-name=GM_aw+gen
#SBATCH --time=06:00:00
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

# [hus, psl, siconc, ta, tas, tos, ua, uas, va, vas, wap, zg]
# imporant: for safety best put everything into '', a separate line and DO NOT use spaces when listing arguments!

# ---- GLOBAL MEAN ----
python -m evaluation.main \
  run_plots='["global_mean"]' \
  members='["member1","member2","member4","member5"]' \
  'out.overwrite=true' \
  plots.global_mean.models='["forced_sst","archesweather"]' \

# ---- BIAS MAPS ----
# python -m evaluation.main \
#   'run_plots=["bias_map"]' \
#   'out.overwrite=true' \
#   'plots.bias_map.coastline_colour=black' \
#   'plots.bias_map.models=["free_run_control","forced_sst","forced_sst_2k","forced_sst_4k"]' \
#   'plots.bias_map.time.use_named=TSTP'

# python -m evaluation.main \
#   'run_plots=["bias_map"]' \
#   'out.overwrite=true' \
#   'plots.bias_map.coastline_colour=black' \
#   'plots.bias_map.models=["free_run_prediction"]' \
#   'plots.bias_map.time.use_named=TSTP'


# ---- SOI ----
# python -m evaluation.main \
#   'run_plots=["soi"]' \
#   'out.overwrite=true' \
#   'plots.soi.models=["free_run_control","forced_sst","forced_sst_2k","forced_sst_4k"]'

# python -m evaluation.main \
#   'run_plots=["soi"]' \
#   'out.overwrite=true' \
#   'plots.soi.models=["free_run_prediction"]' \
#   'plots.soi.time.use_named=TSTP'

# python -m evaluation.main \
#   'run_plots=["soi"]' \
#   'out.overwrite=true' \
#   'plots.soi.models=["free_run_prediction"]' \
#   'plots.soi.time.start="2015-01-01"' \
#   'plots.soi.time.end="2050-12-31"' \
#   'plots.soi.hist_kde.enabled=false'

# python -m evaluation.main \
#   'run_plots=["soi"]' \
#   'out.overwrite=true' \
#   'plots.soi.models=["archesweather"]' \
#   'members=["member1","member2","member4","member5"]'


echo "ALL DONE."