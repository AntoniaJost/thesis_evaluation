# evaluation/metrics/global_mean.py
from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import os
import hydra

from evaluation.general_functions import (
    ensure_allowed_var,
    resolve_period,
    open_model_da,
    open_era5_da,
    ensemble_mean_as_member,
    conversion_rules,
)


def lin_reg(x: xr.DataArray):
    slope, intercept, r, p, std_err = stats.linregress(x.time.dt.year.values, x.values)
    lrg = slope * x.time.dt.year.values + intercept
    return lrg, slope


def trend_decay(slope):
    return round(slope * 10, 3)


def annual_weighted_mean(da_monthly: xr.DataArray) -> xr.DataArray:
    days = da_monthly.time.dt.days_in_month
    return (da_monthly * days).resample(time="1YE").sum() / days.resample(time="1YE").sum()


def area_weighted_global_mean(da: xr.DataArray, lat_name="lat", lon_name="lon") -> xr.DataArray:
    weights = np.cos(np.deg2rad(da[lat_name]))
    return da.weighted(weights).mean(dim=(lon_name, lat_name))


def run(cfg):
    plot_cfg = cfg.plots.global_mean #.metrics.global_mean
    ensure_allowed_var(cfg, plot_cfg.variable)
    start, end = resolve_period(cfg, plot_cfg)

    var = plot_cfg.variable
    meta = cfg.variables.meta.get(var, None)
    long_name = meta.long_name if meta else var
    unit = meta.unit if meta else ""

    # --- model annual GM per run, per member
    models_agm = {}
    for model_name in plot_cfg.models:
        model_cfg = cfg.datasets.models[model_name]
        member_agm = {}
        for m in cfg.members:
            da = open_model_da(model_cfg, cfg, m, var, model_cfg.modelname, plot_cfg.freq, start, end, grid=plot_cfg.grid)
            da = conversion_rules(var, da, cfg)
            gm = area_weighted_global_mean(da)
            agm = annual_weighted_mean(gm)
            member_agm[m] = agm
        models_agm[model_name] = member_agm

    # --- ERA5 annual GM (with optional offsets)
    era5_da = open_era5_da(cfg, var=var, start=start, end=end)
    era5_da = conversion_rules(var, era5_da, cfg)

    agm_era5_by_offset = {}
    lrg_era5_by_offset = {}
    trend_era5_by_offset = {}

    for offset_k in plot_cfg.era5_offsets_k:
        da_off = era5_da + offset_k if offset_k != 0 else era5_da
        gm = area_weighted_global_mean(da_off)
        agm = annual_weighted_mean(gm)
        agm_era5_by_offset[offset_k] = agm
        lrg, slope = lin_reg(agm)
        lrg_era5_by_offset[offset_k] = lrg
        trend_era5_by_offset[offset_k] = trend_decay(slope)

    # --- ensemble stats (min/max/mean)
    minmax_ds = {}
    lrg_mean_ens = {}
    trend_mean_ens = {}

    for model_name, members_dict in models_agm.items():
        member_names = sorted(members_dict.keys())
        da_members = xr.concat([members_dict[m] for m in member_names], dim="member").assign_coords(member=member_names)
        stats_ds = xr.Dataset({"min": da_members.min("member"),
                               "max": da_members.max("member"),
                               "mean": da_members.mean("member")})
        minmax_ds[model_name] = stats_ds
        lrg, slope = lin_reg(stats_ds["mean"])
        lrg_mean_ens[model_name] = lrg
        trend_mean_ens[model_name] = trend_decay(slope)

    # --- plotting (kept close to your style; colours from cfg)
    fig, ax = plt.subplots(figsize=tuple(plot_cfg.figsize), constrained_layout=True)

    # ERA5 (0K) solid + trend dashed
    years_era5 = agm_era5_by_offset[0].time.dt.year.values
    ax.plot(years_era5, agm_era5_by_offset[0].values, color="black", linewidth=1.5,
            label=f"ERA5 (Trend: {trend_era5_by_offset[0]} {plot_cfg.unit_out}/decade)", zorder=5)
    ax.plot(years_era5, lrg_era5_by_offset[0], color="black", linewidth=1.2, linestyle="--", alpha=0.9, zorder=7)

    # extra ERA5 offset trend lines
    for offset_k in plot_cfg.era5_offsets_k:
        if offset_k == 0:
            continue
        ax.plot(years_era5, lrg_era5_by_offset[offset_k], color="black",
                linewidth=1.2, linestyle="--", alpha=0.7, zorder=6)

    # model members thin
    for i, model_name in enumerate(plot_cfg.models):
        c = plot_cfg.colours[i % len(plot_cfg.colours)]
        for m, da in models_agm[model_name].items():
            years = da.time.dt.year.values
            ax.plot(years, da.values, color=c, alpha=0.18, linewidth=0.9, zorder=2)

    # spread + mean + mean trend
    for i, model_name in enumerate(plot_cfg.models):
        c = plot_cfg.colours[i % len(plot_cfg.colours)]
        fill = plot_cfg.colours_light[i % len(plot_cfg.colours_light)]
        ds = minmax_ds[model_name]
        years = ds.time.dt.year.values

        ax.fill_between(years, ds["min"].values, ds["max"].values,
                        color=fill, alpha=0.30,
                        label=f"{model_name} spread (Ens. trend: {trend_mean_ens[model_name]} {plot_cfg.unit_out}/decade)",
                        zorder=1)

        ax.plot(years, ds["mean"].values, color=c, linewidth=1.2, alpha=0.95, zorder=4)
        ax.plot(years, lrg_mean_ens[model_name], color=c, linewidth=1.0, linestyle="--", zorder=5)

    ax.set_title(plot_cfg.title.format(var=var), pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{long_name} ({plot_cfg.unit_out})")

    ax.xaxis.set_major_locator(mticker.MultipleLocator(plot_cfg.ticks.major))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(plot_cfg.ticks.minor))
    ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
    ax.grid(False, which="minor")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.01)

    ax.legend(loc=plot_cfg.legend.loc, frameon=False, fontsize="x-small", borderaxespad=0.0)
    # plt.show()
    if cfg.out.savefig:
        outdir = os.path.join(
            hydra.utils.get_original_cwd(),
            cfg.out.dir
        )
        os.makedirs(outdir, exist_ok=True)

        fname = "gmst.png"
        fig.savefig(
            os.path.join(outdir, fname),
            dpi=cfg.out.dpi,
            bbox_inches="tight"
        )
        plt.close(fig)
    else:
        plt.show()
