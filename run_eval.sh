#!/bin/bash
#SBATCH --job-name=diff_windspeed
#SBATCH --time=03:00:00
#SBATCH --partition=compute
#SBATCH --account=bk1450
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

source ~/.bashrc
conda activate thesis_eval
set -euo pipefail

# ==== RANGE SUMMARY =====
# if running for the first time and bias maps are wanted, calculate the corresponding csv files first! only needed once (depending on the number of models you select, this will take approx. x hours (for 6 models))

# python -m evaluation.range_summary \
#   'range_summary.models_to_process=["free_run_control","free_run_prediction","forced_sst","forced_sst_2k","forced_sst_4k","archesweather"]' \
#   'range_summary.tag=_ALL'

# python -m evaluation.range_plevs \
#   'range_summary.models_to_process=["forced_sst_2k","forced_sst_4k"]' \
#   'range_summary.tag=_sst2+sst4'

# python evaluation/range_windspeed.py --suffix _ALL

# options: ["free_run_control","free_run_prediction","forced_sst","forced_sst_2k","forced_sst_4k","archesweather"]

# ===== CALC PLOTS =====
# variable options: [hus, psl, siconc, ta, tas, tos, ua, uas, va, vas, wap, zg]
# imporant: for safety best to put everything into '', a separate line and DO NOT use spaces when listing arguments!

# ---- GLOBAL MEAN ----
# full period
# python -m evaluation.main \
#   run_plots='["global_mean"]' \
#   'out.overwrite=true' \
#   'plots.global_mean.variable=tas' \
#   'plots.global_mean.plev=850' \
#   'plots.global_mean.models=["free_run_control","free_run_prediction","forced_sst","forced_sst_2k","forced_sst_4k","archesweather"]' \
#   'plots.global_mean.show_era5_offset_trends=true' \
#   'plots.global_mean.legend.inside_plot=false'

# # TRP
# python -m evaluation.main \
#   run_plots='["global_mean"]' \
#   'out.overwrite=true' \
#   'plots.global_mean.variable=tas' \
#   'plots.global_mean.plev=850' \
#   'plots.global_mean.models=["free_run_control","forced_sst","forced_sst_2k","forced_sst_4k","archesweather"]' \
#   'plots.global_mean.time.use_named=TRP' \
#   'plots.global_mean.show_era5_offset_trends=true' \
#   'plots.global_mean.legend.inside_plot=false'
  
# # TSTP
# python -m evaluation.main \
#   run_plots='["global_mean"]' \
#   'out.overwrite=true' \
#   'plots.global_mean.variable=tas' \
#   'plots.global_mean.plev=850' \
#   'plots.global_mean.models=["free_run_control","free_run_prediction","forced_sst","forced_sst_2k","forced_sst_4k","archesweather"]' \
#   'plots.global_mean.time.use_named=TSTP' \
#   'plots.global_mean.show_era5_offset_trends=true' \
#   'plots.global_mean.legend.inside_plot=false' 

# ---- ANOMALIES ----
# python -m evaluation.main \
#   run_plots='["anomalies"]' \
#   'out.overwrite=true' \
#   plots.anomalies.models='["forced_sst","archesweather"]' \

# ---- BIAS MAPS ----
# for 5 models at once, this can take a bit over 12h
# TRP
# python -m evaluation.main \
#   'run_plots=["bias_map"]' \
#   'out.overwrite=true' \
#   'plots.bias_map.variable=tas' \
#   'plots.bias_map.plev=500' \
#   'plots.bias_map.models=["archesweather"]' \
#   'plots.bias_map.time.use_named=TRP' \
#   'plots.bias_map.ticks_everyX_model=1' \
#   'plots.bias_map.keep_0_tick_diff=true' \
#   'plots.bias_map.range_source.suffix="_sst0+AW+sst2+sst4"' \
#   'plots.bias_map.range_source.percentile=99' \
#   'plots.bias_map.range_source.csv_file1="outputs/range_summary/old_WORKING/range_summary_compact${.suffix}.csv"' \
#   'plots.bias_map.range_source.csv_file2="outputs/range_summary/old_WORKING/model_minus_era5_summary_by_var_plev${.suffix}.csv"' \
#   'plots.bias_map.special_outdir="tas"' 

# # TSTP
# python -m evaluation.main \
#   'run_plots=["bias_map"]' \
#   'out.overwrite=true' \
#   'plots.bias_map.variable=zg' \
#   'plots.bias_map.plev=500' \
#   'plots.bias_map.models=["free_run_control","free_run_prediction","forced_sst","forced_sst_2k","forced_sst_4k","archesweather"]' \
#   'plots.bias_map.time.use_named=TSTP' \
#   'plots.bias_map.ticks_everyX_model=1' \
#   'plots.bias_map.keep_0_tick_diff=true' \
#   'plots.bias_map.range_source.suffix="_sst0+AW+sst2+sst4"' \
#   'plots.bias_map.range_source.percentile=99'

# # full period
# python -m evaluation.main \
#   'run_plots=["bias_map"]' \
#   'out.overwrite=true' \
#   'plots.bias_map.variable=zg' \
#   'plots.bias_map.plev=500' \
#   'plots.bias_map.models=["free_run_control","forced_sst","forced_sst_2k","forced_sst_4k","archesweather"]' \
#   'plots.bias_map.ticks_everyX_model=1' \
#   'plots.bias_map.keep_0_tick_diff=true' \
#   'plots.bias_map.range_source.suffix="_sst0+AW+sst2+sst4"' \
#   'plots.bias_map.range_source.percentile=99'

# ---- DIFFERENCE MAPS WITH RAW VALUES ----
# TRP
# python -m evaluation.main \
#   'run_plots=["diff_map_raw"]' \
#   'out.overwrite=true' \
#   'plots.diff_map_raw.variable=uas' \
#   'plots.diff_map_raw.plev=500' \
#   'plots.diff_map_raw.models=["forced_sst"]' \
#   'plots.diff_map_raw.time.use_named=TRP' \
#   'plots.diff_map_raw.ticks_everyX_model=2' \
#   'plots.diff_map_raw.keep_0_tick_diff=true' \
#   'plots.diff_map_raw.global_centre=0' \
#   'plots.diff_map_raw.cmap_model=bwr' \
#   'plots.diff_map_raw.detrend.enabled=false' \
#   'plots.diff_map_raw.detrend.preserve_mean=false' \
#   'plots.diff_map_raw.special_outdir="thesis"' \
#   'plots.diff_map_raw.range_source.suffix="_sst0+AW"' \
#   'plots.diff_map_raw.range_source.csv_file1="outputs/range_summary/range_summary_compact${.suffix}.csv"' \
#   'plots.diff_map_raw.range_source.csv_file2="outputs/range_summary/model_minus_era5_summary_by_var_plev${.suffix}.csv"' \
#   'plots.diff_map_raw.range_source.percentile=99'

# # TSTP
# python -m evaluation.main \
#   'run_plots=["diff_map_raw"]' \
#   'out.overwrite=true' \
#   'plots.diff_map_raw.variable=uas' \
#   'plots.diff_map_raw.plev=500' \
#   'plots.diff_map_raw.models=["forced_sst"]' \
#   'plots.diff_map_raw.time.use_named=TSTP' \
#   'plots.diff_map_raw.ticks_everyX_model=2' \
#   'plots.diff_map_raw.keep_0_tick_diff=true' \
#   'plots.diff_map_raw.global_centre=0' \
#   'plots.diff_map_raw.cmap_model=bwr' \
#   'plots.diff_map_raw.detrend.enabled=false' \
#   'plots.diff_map_raw.detrend.preserve_mean=false' \
#   'plots.diff_map_raw.special_outdir="thesis"' \
#   'plots.diff_map_raw.range_source.suffix="_sst0+AW"' \
#   'plots.diff_map_raw.range_source.csv_file1="outputs/range_summary/range_summary_compact${.suffix}.csv"' \
#   'plots.diff_map_raw.range_source.csv_file2="outputs/range_summary/model_minus_era5_summary_by_var_plev${.suffix}.csv"' \
#   'plots.diff_map_raw.range_source.percentile=99'

# # full period
# python -m evaluation.main \
#   'run_plots=["diff_map_raw"]' \
#   'out.overwrite=true' \
#   'plots.diff_map_raw.variable=uas' \
#   'plots.diff_map_raw.plev=500' \
#   'plots.diff_map_raw.models=["forced_sst"]' \
#   'plots.diff_map_raw.time.use_named=null' \
#   'plots.diff_map_raw.ticks_everyX_model=2' \
#   'plots.diff_map_raw.keep_0_tick_diff=true' \
#   'plots.diff_map_raw.global_centre=0' \
#   'plots.diff_map_raw.cmap_model=bwr' \
#   'plots.diff_map_raw.detrend.enabled=false' \
#   'plots.diff_map_raw.detrend.preserve_mean=false' \
#   'plots.diff_map_raw.special_outdir="thesis"' \
#   'plots.diff_map_raw.range_source.suffix="_sst0+AW"' \
#   'plots.diff_map_raw.range_source.csv_file1="outputs/range_summary/range_summary_compact${.suffix}.csv"' \
#   'plots.diff_map_raw.range_source.csv_file2="outputs/range_summary/model_minus_era5_summary_by_var_plev${.suffix}.csv"' \
#   'plots.diff_map_raw.range_source.percentile=raw'

# ---- SOI ----
# runs within minutes for all models at once
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

# ---- INDIVIDUAL PLOTS ----
# # full period
# python -m evaluation.main \
# run_plots='["individual_plots"]' \
# out.overwrite=true \
# members=[member1] \
# plots.individual_plots.variable="uas" \
# plots.individual_plots.plev="850" \
# plots.individual_plots.models='["forced_sst"]' \
# plots.individual_plots.map_era5=true \
# plots.individual_plots.method=map \
# plots.individual_plots.time_stat=raw \
# plots.individual_plots.detrend.enabled=false \
# plots.individual_plots.detrend.preserve_mean=true \
# plots.individual_plots.difference=true \
# plots.individual_plots.anomaly=false \
# plots.individual_plots.special_outdir="thesis" \
# plots.individual_plots.include_ensemble_mean_as_member=true \
# plots.individual_plots.only_mean=false \
# plots.individual_plots.location=global \
# plots.individual_plots.global_centre=0 \
# plots.individual_plots.colourbar.tick_every=2 \
# plots.individual_plots.colour_scheme=BrBG \
# plots.individual_plots.range_source.suffix="_sst0+AW" \
# plots.individual_plots.range_source.percentile=99 

# python -m evaluation.main \
# run_plots='["individual_plots"]' \
# out.overwrite=true \
# plots.individual_plots.variable="hus" \
# plots.individual_plots.plev="850" \
# plots.individual_plots.models='["free_run_control","forced_sst","forced_sst_2k","forced_sst_4k","archesweather"]' \
# plots.individual_plots.map_era5=false \
# plots.individual_plots.method=map \
# plots.individual_plots.time_stat=trend \
# plots.individual_plots.detrend.enabled=false \
# plots.individual_plots.detrend.preserve_mean=true \
# plots.individual_plots.difference=true \
# plots.individual_plots.anomaly=false \
# plots.individual_plots.special_outdir="hus@850hPa/full_period/difference" \
# plots.individual_plots.only_mean=true \
# plots.individual_plots.location=global \
# plots.individual_plots.global_centre=0 \
# plots.individual_plots.colourbar.tick_every=2 \
# plots.individual_plots.colour_scheme=BrBG \
# plots.individual_plots.range_source.suffix="_ALL" \
# plots.individual_plots.range_source.percentile=99

# # TRP
# python -m evaluation.main \
# run_plots='["individual_plots"]' \
# out.overwrite=true \
# plots.individual_plots.variable="hus" \
# plots.individual_plots.plev="850" \
# plots.individual_plots.time.use_named=TRP \
# plots.individual_plots.models='["free_run_control","forced_sst","forced_sst_2k","forced_sst_4k","archesweather"]' \
# plots.individual_plots.map_era5=true \
# plots.individual_plots.method=map \
# plots.individual_plots.time_stat=trend \
# plots.individual_plots.detrend.enabled=false \
# plots.individual_plots.detrend.preserve_mean=true \
# plots.individual_plots.difference=false \
# plots.individual_plots.anomaly=false \
# plots.individual_plots.special_outdir="hus@850hPa/TRP" \
# plots.individual_plots.only_mean=true \
# plots.individual_plots.location=global \
# plots.individual_plots.global_centre=0 \
# plots.individual_plots.colourbar.tick_every=3 \
# plots.individual_plots.colour_scheme=bwr \
# plots.individual_plots.range_source.suffix="_ALL" \
# plots.individual_plots.range_source.percentile=99 

# python -m evaluation.main \
# run_plots='["individual_plots"]' \
# out.overwrite=true \
# plots.individual_plots.variable="hus" \
# plots.individual_plots.plev="850" \
# plots.individual_plots.time.use_named=TRP \
# plots.individual_plots.models='["free_run_control","forced_sst","forced_sst_2k","forced_sst_4k","archesweather"]' \
# plots.individual_plots.map_era5=false \
# plots.individual_plots.method=map \
# plots.individual_plots.time_stat=trend \
# plots.individual_plots.detrend.enabled=false \
# plots.individual_plots.detrend.preserve_mean=true \
# plots.individual_plots.difference=true \
# plots.individual_plots.anomaly=false \
# plots.individual_plots.special_outdir="hus@850hPa/TRP/difference" \
# plots.individual_plots.only_mean=true \
# plots.individual_plots.location=global \
# plots.individual_plots.global_centre=0 \
# plots.individual_plots.colourbar.tick_every=2 \
# plots.individual_plots.colour_scheme=BrBG \
# plots.individual_plots.range_source.suffix="_ALL" \
# plots.individual_plots.range_source.percentile=99

# # TSTP
# python -m evaluation.main \
# run_plots='["individual_plots"]' \
# out.overwrite=true \
# plots.individual_plots.variable="hus" \
# plots.individual_plots.plev="850" \
# plots.individual_plots.time.use_named=TSTP \
# plots.individual_plots.models='["free_run_control","free_run_prediction","forced_sst","forced_sst_2k","forced_sst_4k","archesweather"]' \
# plots.individual_plots.map_era5=true \
# plots.individual_plots.method=map \
# plots.individual_plots.time_stat=trend \
# plots.individual_plots.detrend.enabled=false \
# plots.individual_plots.detrend.preserve_mean=true \
# plots.individual_plots.difference=false \
# plots.individual_plots.anomaly=false \
# plots.individual_plots.special_outdir="hus@850hPa/TSTP" \
# plots.individual_plots.only_mean=true \
# plots.individual_plots.location=global \
# plots.individual_plots.global_centre=0 \
# plots.individual_plots.colourbar.tick_every=3 \
# plots.individual_plots.colour_scheme=bwr \
# plots.individual_plots.range_source.suffix="_ALL" \
# plots.individual_plots.range_source.percentile=99 

# python -m evaluation.main \
# run_plots='["individual_plots"]' \
# out.overwrite=true \
# plots.individual_plots.variable="hus" \
# plots.individual_plots.plev="850" \
# plots.individual_plots.time.use_named=TSTP \
# plots.individual_plots.models='["free_run_control","free_run_prediction","forced_sst","forced_sst_2k","forced_sst_4k","archesweather"]' \
# plots.individual_plots.map_era5=false \
# plots.individual_plots.method=map \
# plots.individual_plots.time_stat=trend \
# plots.individual_plots.detrend.enabled=false \
# plots.individual_plots.detrend.preserve_mean=true \
# plots.individual_plots.difference=true \
# plots.individual_plots.anomaly=false \
# plots.individual_plots.special_outdir="hus@850hPa/TSTP/difference" \
# plots.individual_plots.only_mean=true \
# plots.individual_plots.location=global \
# plots.individual_plots.global_centre=0 \
# plots.individual_plots.colourbar.tick_every=2 \
# plots.individual_plots.colour_scheme=BrBG \
# plots.individual_plots.range_source.suffix="_ALL" \
# plots.individual_plots.range_source.percentile=99

# ---- ZONAL MEAN ----
# # full period
# python -m evaluation.main \
# run_plots='["zonal_mean"]' \
# out.overwrite=true \
# members=[member1] \
# plots.zonal_mean.variable="[ua]" \
# plots.zonal_mean.models='["forced_sst"]' \
# plots.zonal_mean.map_era5=true \
# plots.zonal_mean.all_single_plots=true \
# plots.zonal_mean.difference=true \
# plots.zonal_mean.include_ensemble_mean_as_member=false \
# plots.zonal_mean.only_mean=false \
# plots.zonal_mean.cmap_absolute=bwr \
# plots.zonal_mean.cmap_difference=BrBG \
# plots.zonal_mean.colourbar.suffix="_sst0+AW" \
# plots.zonal_mean.colourbar.percentile=99 \
# plots.zonal_mean.colourbar.target_bins=20 \
# plots.zonal_mean.colourbar.tick_every=2 \
# plots.zonal_mean.special_outdir="thesis"

# # TRP
# python -m evaluation.main \
# run_plots='["zonal_mean"]' \
# out.overwrite=true \
# plots.zonal_mean.variable="[ta, ua, va, wap, hus, zg]" \
# plots.zonal_mean.time.use_named=TRP \
# plots.zonal_mean.models='["free_run_control","forced_sst","forced_sst_2k","forced_sst_4k","archesweather"]' \
# plots.zonal_mean.map_era5=true \
# plots.zonal_mean.all_single_plots=false \
# plots.zonal_mean.difference=true \
# plots.zonal_mean.only_mean=false \
# plots.zonal_mean.cmap_absolute=bwr \
# plots.zonal_mean.cmap_difference=BrBG \
# plots.zonal_mean.colourbar.suffix="_ALL" \
# plots.zonal_mean.colourbar.percentile=99 \
# plots.zonal_mean.colourbar.target_bins=20 \
# plots.zonal_mean.colourbar.tick_every=2 \
# plots.zonal_mean.special_outdir="ALL_99"

# # TSTP
# python -m evaluation.main \
# run_plots='["zonal_mean"]' \
# out.overwrite=true \
# plots.zonal_mean.variable="[ta, ua, va, wap, hus, zg]" \
# plots.zonal_mean.time.use_named=TSTP \
# plots.zonal_mean.models='["free_run_control","free_run_prediction","forced_sst","forced_sst_2k","forced_sst_4k","archesweather"]' \
# plots.zonal_mean.map_era5=true \
# plots.zonal_mean.all_single_plots=false \
# plots.zonal_mean.difference=true \
# plots.zonal_mean.only_mean=false \
# plots.zonal_mean.cmap_absolute=bwr \
# plots.zonal_mean.cmap_difference=BrBG \
# plots.zonal_mean.colourbar.suffix="_ALL" \
# plots.zonal_mean.colourbar.percentile=99 \
# plots.zonal_mean.colourbar.target_bins=20 \
# plots.zonal_mean.colourbar.tick_every=2 \
# plots.zonal_mean.special_outdir="ALL_99"

# ---- WIND SPEED ----
python -m evaluation.main \
run_plots='["wind"]' \
out.overwrite=true \
plots.wind.plev='[surface, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000]' \
plots.wind.models='["forced_sst"]' \
plots.wind.map_era5=true \
plots.wind.time.use_named=null \
plots.wind.difference=true \
plots.wind.include_ensemble_mean_as_member=true \
plots.wind.only_mean=true \
plots.wind.location=global \
plots.wind.colour_scheme=Blues \
plots.wind.colour_diff=BrBG \
plots.wind.special_outdir="thesis" \
plots.wind.range_source.suffix="_sst0+AW" \
plots.wind.range_source.percentile=99 


echo "ALL DONE."