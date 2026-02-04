# evaluation/metrics/anomalies.py
from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

from evaluation.io import ensure_allowed_var, resolve_period
from evaluation.metrics.global_mean import run as run_global_mean  # optional reuse
from evaluation.metrics.global_mean import lin_reg, trend_decay


def to_anomaly(da: xr.DataArray, baseline_start: str, baseline_end: str):
    base = da.sel(time=slice(baseline_start, baseline_end)).mean("time")
    return da - base, float(base.values)


def run(cfg):
    plot_cfg = cfg.plots #.metrics.anomalies
    ensure_allowed_var(cfg, plot_cfg.variable)

    # We expect global_mean plot config already describes models/freq/etc.
    # To keep minimal changes, we replicate the relevant compute logic from global_mean.
    from evaluation.metrics.global_mean import annual_weighted_mean, area_weighted_global_mean
    from evaluation.io import open_model_da, open_era5_da, maybe_convert_units

    start, end = resolve_period(cfg, plot_cfg)
    var = plot_cfg.variable

    # compute ERA5 annual GM
    era5_da = maybe_convert_units(var, open_era5_da(cfg, var, start, end), cfg)
    gm_era5 = area_weighted_global_mean(era5_da)
    agm_era5 = annual_weighted_mean(gm_era5)

    # anomalies relative to baseline
    base_start = plot_cfg.baseline.start
    base_end = plot_cfg.baseline.end
    anom_era5, _ = to_anomaly(agm_era5, base_start, base_end)
    lrg_era5_anom, slope_era5 = lin_reg(anom_era5)
    trend_era5 = trend_decay(slope_era5)

    # model members
    anom_members = {}
    minmax_ds = {}
    lrg_mean_ens = {}
    trend_mean_ens = {}

    for model_name in plot_cfg.models:
        model_cfg = cfg.datasets.models[model_name]
        member_anom = {}
        for m in cfg.members:
            da = maybe_convert_units(var, open_model_da(model_cfg, m, var, plot_cfg.freq, start, end, grid=plot_cfg.grid), cfg)
            gm = area_weighted_global_mean(da)
            agm = annual_weighted_mean(gm)
            anom, _ = to_anomaly(agm, base_start, base_end)
            member_anom[m] = anom
        anom_members[model_name] = member_anom

        member_names = sorted(member_anom.keys())
        da_members = xr.concat([member_anom[m] for m in member_names], dim="member").assign_coords(member=member_names)

        stats_ds = xr.Dataset({"min": da_members.min("member"),
                               "max": da_members.max("member"),
                               "mean": da_members.mean("member")})
        minmax_ds[model_name] = stats_ds

        lrg, slope = lin_reg(stats_ds["mean"])
        lrg_mean_ens[model_name] = lrg
        trend_mean_ens[model_name] = trend_decay(slope)

    # plot
    fig, ax = plt.subplots(figsize=tuple(plot_cfg.figsize))

    years = anom_era5.time.dt.year.values
    ax.plot(years, anom_era5.values, color="black", linewidth=1.5,
            label=f"ERA5 (Trend = {trend_era5} {plot_cfg.unit_out}/dec)", zorder=6)
    ax.plot(years, lrg_era5_anom, color="black", linestyle="--", linewidth=1.0, zorder=7)

    for i, model_name in enumerate(plot_cfg.models):
        c = plot_cfg.colours[i % len(plot_cfg.colours)]
        for m, anom in anom_members[model_name].items():
            yrs = anom.time.dt.year.values
            ax.plot(yrs, anom.values, color=c, alpha=0.20, linewidth=0.8, zorder=2)

    for i, model_name in enumerate(plot_cfg.models):
        c = plot_cfg.colours[i % len(plot_cfg.colours)]
        fill = plot_cfg.colours_light[i % len(plot_cfg.colours_light)]
        ds = minmax_ds[model_name]
        yrs = ds.time.dt.year.values

        ax.fill_between(yrs, ds["min"].values, ds["max"].values,
                        color=fill, alpha=0.25, zorder=1,
                        label=f"{model_name} range (Trend = {trend_mean_ens[model_name]} {plot_cfg.unit_out}/dec)")
        ax.plot(yrs, ds["mean"].values, color=c, linewidth=1.2, zorder=4)
        ax.plot(yrs, lrg_mean_ens[model_name], color=c, linestyle="--", linewidth=1.0, zorder=5)

    ax.axhline(0, color="0.3", linewidth=1.0, alpha=0.35)
    ax.set_title(plot_cfg.title.format(base_start=base_start[:7], base_end=base_end[:7]), fontsize=12, pad=8)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Anomaly ({plot_cfg.unit_out})")

    ax.xaxis.set_major_locator(mticker.MultipleLocator(plot_cfg.ticks.major))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(plot_cfg.ticks.minor))
    ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.5, alpha=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.01)

    ax.legend(loc=plot_cfg.legend.loc, fontsize=9, frameon=False, handlelength=2.6, borderaxespad=0.6)
    plt.show()
