#!/bin/bash
#SBATCH --job-name=wind
#SBATCH --time=05:00:00
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
python -m evaluation.main \
  run_plots='["global_mean"]' \
  'out.overwrite=ask' \
  'plots.global_mean.freq="monthly"' \
  'plots.global_mean.time.use_named=null' \
  'plots.global_mean.variable=ua' \
  'plots.global_mean.plev=250' \
  'plots.global_mean.models=["forced_sst"]' \
  'plots.global_mean.show_era5_offset_trends=false' \
  'plots.global_mean.legend.inside_plot=true' \
  'plots.global_mean.special_outdir="thesis"'

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
#   out.overwrite=ask \
#   plots.anomalies.variable=ua \
#   plots.anomalies.plev=300 \
#   plots.anomalies.models='["forced_sst"]' \
#   plots.anomalies.time.use_named=null \
#   plots.anomalies.mode=detrend \
#   plots.anomalies.detrend.preserve_mean=false \
#   plots.anomalies.legend.inside_plot=true \
#   plots.anomalies.special_outdir="test"


# ---- BIAS MAPS ----
# for 5 models at once, this can take a bit over 12h
# TRP
# python -m evaluation.main \
#   'run_plots=["bias_map"]' \
#   'out.overwrite=true' \
#   'plots.bias_map.variable=hus' \
#   'plots.bias_map.plev=500' \
#   'plots.bias_map.models=["archesweather"]' \
#   'plots.bias_map.time.use_named=TRP' \
#   'plots.bias_map.ticks_everyX_model=1' \
#   'plots.bias_map.keep_0_tick_diff=true' \
#   'plots.bias_map.bottom_numbers_decimals=2' \
#   'plots.bias_map.range_source.suffix="_sst0+AW+sst2+sst4"' \
#   'plots.bias_map.range_source.percentile=99' \
#   'plots.bias_map.range_source.csv_file1="outputs/range_summary/old_WORKING/range_summary_compact${.suffix}.csv"' \
#   'plots.bias_map.range_source.csv_file2="outputs/range_summary/old_WORKING/model_minus_era5_summary_by_var_plev${.suffix}.csv"' \
#   'plots.bias_map.special_outdir="test"' 

# # TSTP
# python -m evaluation.main \
#   'run_plots=["bias_map"]' \
#   'out.overwrite=true' \
#   'plots.bias_map.variable=zg' \
#   'plots.bias_map.plev=500' \
#   'plots.bias_map.models=["free_run_control","free_run_prediction","forced_sst","forced_sst_2k","forced_sst_4k","archesweather"]' \
#   'plots.bias_map.time.use_named=TSTP' \
#   'plots.bias_map.ticks_everyX_model=1' \
#   'plots.bias_map.bottom_numbers_decimals=2' \
#   'plots.bias_map.keep_0_tick_diff=true' \
#   'plots.bias_map.range_source.suffix="_sst0+AW+sst2+sst4"' \
#   'plots.bias_map.range_source.percentile=99'

# # full period
python -m evaluation.main \
  'run_plots=["bias_map"]' \
  'out.overwrite=ask' \
  'plots.bias_map.variable=tas' \
  'plots.bias_map.plev=850' \
  'plots.bias_map.models=["forced_sst"]' \
  'plots.bias_map.time.use_named=TSTP' \
  'plots.bias_map.include_ensemble_mean_as_member=true' \
  'plots.bias_map.cmap_model=bwr' \
  'plots.bias_map.bottom_numbers_decimals=2' \
  'plots.bias_map.ticks_everyX_model=2' \
  'plots.bias_map.ticks_everyX_diff=3' \
  'plots.bias_map.keep_0_tick_diff=true' \
  'plots.bias_map.range_source.suffix="_sst0+AW"' \
  'plots.bias_map.range_source.percentile=99' \
  'plots.bias_map.special_outdir="thesis"' 

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
python -m evaluation.main \
  'run_plots=["diff_map_raw"]' \
  'out.overwrite=ask' \
  'plots.diff_map_raw.variable=tos' \
  'plots.diff_map_raw.plev=850' \
  'plots.diff_map_raw.models=["forced_sst"]' \
  'plots.diff_map_raw.time.use_named=TSTP' \
  'plots.diff_map_raw.season=full' \
  'plots.diff_map_raw.colourbar.ticks_everyX_model=2' \
  'plots.diff_map_raw.colourbar.ticks_everyX_diff=2' \
  'plots.diff_map_raw.colourbar.keep_0_tick_diff=true' \
  'plots.diff_map_raw.global_centre=0' \
  'plots.diff_map_raw.colourbar.cmap_model=bwr' \
  'plots.diff_map_raw.bottom_numbers_decimals=2' \
  'plots.diff_map_raw.detrend.enabled=false' \
  'plots.diff_map_raw.detrend.preserve_mean=false' \
  'plots.diff_map_raw.special_outdir="tos"' \
  'plots.diff_map_raw.range_source.suffix="_sst0+AW"' \
  'plots.diff_map_raw.range_source.percentile=99'

# ---- SOI ----
# runs within minutes for all models at once
# python -m evaluation.main \
#   'run_plots=["soi"]' \
#   'out.overwrite=true' \
#   'plots.soi.models=["free_run_control","forced_sst","forced_sst_2k","forced_sst_4k"]'

python -m evaluation.main \
  'run_plots=["soi"]' \
  'out.overwrite=true' \
  'plots.soi.models=["forced_sst"]' \
  'plots.soi.time.use_named=null' \
  'plots.soi.special_outdir="thesis"'

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
python -m evaluation.main \
run_plots='["individual_plots"]' \
out.overwrite=ask \
members=[member1] \
plots.individual_plots.freq=monthly \
plots.individual_plots.time.use_named=null \
plots.individual_plots.time.start="1979-01-01" \
plots.individual_plots.time.end="2024-12-31" \
plots.individual_plots.season=[full] \
plots.individual_plots.variable="tas" \
plots.individual_plots.plev="[250]" \
plots.individual_plots.models='["forced_sst"]' \
plots.individual_plots.map_era5=true \
plots.individual_plots.method=map \
plots.individual_plots.time_stat=raw \
plots.individual_plots.detrend.enabled=false \
plots.individual_plots.detrend.preserve_mean=false \
plots.individual_plots.difference=false \
plots.individual_plots.anomaly=false \
plots.individual_plots.baseline.start="1981-01-01" \
plots.individual_plots.baseline.end="2010-12-31" \
plots.individual_plots.special_outdir="thesis" \
plots.individual_plots.include_ensemble_mean_as_member=true \
plots.individual_plots.only_mean=true \
plots.individual_plots.location=global \
plots.individual_plots.individual.lat0=0 \
plots.individual_plots.individual.lat1=-25 \
plots.individual_plots.individual.lon0=110 \
plots.individual_plots.individual.lon1=240 \
plots.individual_plots.polar.min_latitude=30 \
plots.individual_plots.polar.max_latitude=-30 \
plots.individual_plots.draw_soiBox=false \
plots.individual_plots.global_centre=180 \
plots.individual_plots.central_latitude=-90 \
plots.individual_plots.colourbar.tick_every=2 \
plots.individual_plots.colour_scheme=bwr \
plots.individual_plots.diff_colour=BrBG \
plots.individual_plots.range_source.suffix="_sst0+AW" \
plots.individual_plots.range_source.percentile=99 

# plots.individual_plots.colourbar.manual_vmin=-50 \
# plots.individual_plots.colourbar.manual_vmax=50

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
python -m evaluation.main \
run_plots='["zonal_mean"]' \
out.overwrite=ask \
members=[member1] \
plots.zonal_mean.variable="[hus]" \
plots.zonal_mean.time.use_named=null \
plots.zonal_mean.season=["DJF","JJA"] \
plots.zonal_mean.region=global \
plots.zonal_mean.individual.lat0=0 \
plots.zonal_mean.individual.lat1=-90 \
plots.zonal_mean.individual.lon0=300 \
plots.zonal_mean.individual.lon1=30 \
plots.zonal_mean.models='["forced_sst"]' \
plots.zonal_mean.map_era5=true \
plots.zonal_mean.all_single_plots=false \
plots.zonal_mean.difference=true \
plots.zonal_mean.include_ensemble_mean_as_member=false \
plots.zonal_mean.only_mean=false \
plots.zonal_mean.cmap_absolute=bwr \
plots.zonal_mean.cmap_difference=BrBG \
plots.zonal_mean.colourbar.suffix="_sst0+AW" \
plots.zonal_mean.colourbar.percentile=99 \
plots.zonal_mean.colourbar.target_bins=20 \
plots.zonal_mean.colourbar.tick_every=2 \
plots.zonal_mean.special_outdir="thesis" \
plots.zonal_mean.convert_hus_bounds=false \
plots.zonal_mean.colourbar.manual_vmin=-0.2 \
plots.zonal_mean.colourbar.manual_vmax=0.2 \
plots.zonal_mean.plev=[20000,25000,30000,40000,50000,60000,70000,85000,92500,100000] 


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
out.overwrite=ask \
members='[member1]' \
plots.wind.freq='monthly' \
plots.wind.plev='[850]' \
plots.wind.models='["forced_sst"]' \
plots.wind.background=pressure \
plots.wind.map_era5=true \
plots.wind.time.use_named=null \
plots.wind.time.start="1979-01-01" \
plots.wind.time.end="2024-12-31" \
plots.wind.season=DJF \
plots.wind.difference=false \
plots.wind.include_ensemble_mean_as_member=false \
plots.wind.only_mean=false \
plots.wind.location=global \
plots.wind.global_centre=0 \
plots.wind.central_latitude=60 \
plots.wind.skip=13 \
plots.wind.scale=200 \
plots.wind.q_ref=10 \
plots.wind.colour_speed=Blues \
plots.wind.colour_diff=BrBG \
plots.wind.colourbar.manual=true \
plots.wind.colourbar.manual_vmin=98000 \
plots.wind.colourbar.manual_vmax=102500 \
plots.wind.special_outdir="thesis" \
plots.wind.range_source.suffix="_sst0+AW" \
plots.wind.range_source.percentile=99 

# plots.wind.colourbar.manual=true \
# plots.wind.colourbar.manual_vmin=0 \
# plots.wind.colourbar.manual_vmax=60 

# ---- SEASONAL CYCLE ----
python -m evaluation.main \
  run_plots='["seasonal_cycle"]' \
  out.overwrite=ask \
  plots.seasonal_cycle.variable='["ua"]' \
  plots.seasonal_cycle.plev='[250]' \
  plots.seasonal_cycle.models='["forced_sst"]' \
  plots.seasonal_cycle.time.use_named=null \
  plots.seasonal_cycle.season='["full"]' \
  plots.seasonal_cycle.region=tropics \
  plots.seasonal_cycle.include_ensemble_mean_as_member=true \
  plots.seasonal_cycle.only_mean=true \
  plots.seasonal_cycle.legend.inside_plot=false \
  plots.seasonal_cycle.special_outdir='thesis/23.5'

#   python -m evaluation.main \
#   run_plots='["seasonal_cycle"]' \
#   out.overwrite=true \
#   plots.seasonal_cycle.variable='["ta"]' \
#   plots.seasonal_cycle.plev='[50, 200, 500, 850]' \
#   plots.seasonal_cycle.models='["forced_sst"]' \
#   plots.seasonal_cycle.time.use_named=null \
#   plots.seasonal_cycle.season='["full"]' \
#   plots.seasonal_cycle.region=nothern \
#   plots.seasonal_cycle.include_ensemble_mean_as_member=true \
#   plots.seasonal_cycle.only_mean=true \
#   plots.seasonal_cycle.legend.inside_plot=false \
#   plots.seasonal_cycle.special_outdir='thesis'

# python -m evaluation.main \
#   run_plots='["seasonal_cycle"]' \
#   out.overwrite=true \
#   plots.seasonal_cycle.variable='["ta"]' \
#   plots.seasonal_cycle.plev='[50, 200, 500, 850]' \
#   plots.seasonal_cycle.models='["forced_sst"]' \
#   plots.seasonal_cycle.time.use_named=null \
#   plots.seasonal_cycle.season='["full"]' \
#   plots.seasonal_cycle.region=tropics \
#   plots.seasonal_cycle.include_ensemble_mean_as_member=true \
#   plots.seasonal_cycle.only_mean=true \
#   plots.seasonal_cycle.legend.inside_plot=false \
#   plots.seasonal_cycle.special_outdir='thesis'

# ---- SEASONAL CYCLE FOR ALL MODELS SEPARATELY (YEARLY CYCLE) ----
python -m evaluation.main \
  run_plots='["yearly_cycle"]' \
  out.overwrite=ask \
  plots.yearly_cycle.variable='["tas"]' \
  plots.yearly_cycle.plev='[200, 300, 850]' \
  plots.yearly_cycle.freq=monthly \
  plots.yearly_cycle.models='["forced_sst"]' \
  plots.yearly_cycle.time.use_named=null \
  plots.yearly_cycle.season='["full"]' \
  plots.yearly_cycle.include_ensemble_mean_as_member=true \
  plots.yearly_cycle.only_mean=true \
  plots.yearly_cycle.map_era5=true \
  plots.yearly_cycle.region=individual \
  plots.yearly_cycle.individual.lat0=-20\
  plots.yearly_cycle.individual.lon0=-155 \
  plots.yearly_cycle.individual.lat1=-10 \
  plots.yearly_cycle.individual.lon1=145 \
  plots.yearly_cycle.polar.min_latitude=60 \
  plots.yearly_cycle.polar.max_latitude=-60 \
  plots.yearly_cycle.anomaly=false \
  plots.yearly_cycle.difference=false \
  plots.yearly_cycle.legend.inside_plot=false \
  plots.yearly_cycle.special_outdir='test'


echo "ALL DONE."