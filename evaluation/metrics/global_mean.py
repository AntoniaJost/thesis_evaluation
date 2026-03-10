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
    normalise_plevs,
    open_model_da,
    open_era5_da,
    # ensemble_mean_as_member,
    conversion_rules,
)


def lin_reg(x: xr.DataArray):
    slope, intercept, r, p, std_err = stats.linregress(x.time.dt.year.values, x.values)
    lrg = slope * x.time.dt.year.values + intercept
    return lrg, slope


def trend_decay(slope):
    return round(slope * 10, 3) # °C/decade if slope is °C/year


def annual_weighted_mean(da_monthly: xr.DataArray) -> xr.DataArray:
    # annual mean weighted by days in month (handles leap years).
    days = da_monthly.time.dt.days_in_month
    return (da_monthly * days).resample(time="1YE").sum() / days.resample(time="1YE").sum()


def area_weighted_global_mean(da: xr.DataArray, lat_name: str = "lat", lon_name: str = "lon") -> xr.DataArray:
    weights = np.cos(np.deg2rad(da[lat_name]))
    return da.weighted(weights).mean(dim=(lon_name, lat_name))


# the following function has been written by ChatGPT
def label_line_along_slope(
    ax,
    x,
    y,
    text,
    xpos=0.85,
    angle_boost=4,
    xshift=1.2,
    yshift=-0.1,
    fontsize=9,
    color="black",
    alpha=0.8,
):
    """
    Place a label along a line, rotated to match its slope (with optional boost).
    """
    i = int(len(x) * xpos)

    dx = x[i] - x[i - 1]
    dy = y[i] - y[i - 1]

    angle = np.degrees(np.arctan2(dy, dx)) * angle_boost

    ax.text(
        x[i] + xshift,
        y[i] + yshift,
        text,
        fontsize=fontsize,
        color=color,
        alpha=alpha,
        rotation=angle,
        rotation_mode="anchor",
        ha="left",
        va="center",
    )


def select_colour(model_name, plot_cfg):
    return plot_cfg.colours.base_colours[model_name]


def select_light_colour(model_name, plot_cfg):
    return plot_cfg.colours.colours_light[model_name]


def model_abbrev(name: str) -> str:
    return {
        "forced_sst": "sst0K",
        "forced_sst_2k": "sst2K",
        "forced_sst_4k": "sst4K",
        "free_run_control": "FRc",
        "free_run_prediction": "FR15",
    }.get(name, name)


def run(cfg):
    plot_cfg = cfg.plots.global_mean
    ensure_allowed_var(cfg, plot_cfg.variable)
    start, end = resolve_period(cfg, plot_cfg)

    var = plot_cfg.variable
    plevs = normalise_plevs(getattr(plot_cfg, "plev", None))
    meta = cfg.variables.meta.get(var, None)
    long_name = meta.long_name if meta else var
    unit = meta.unit if meta else ""

    for plev in plevs:
        # --- model annual global mean per run, per member
        models_agm = {}
        for model_name in plot_cfg.models:
            model_cfg = cfg.datasets.models[model_name]
            member_agm = {}
            for m in cfg.members:
                da = open_model_da(model_cfg, cfg, m, var, model_cfg.modelname, plot_cfg.freq, start, end, grid=plot_cfg.grid, plev=plev)
                da, _ = conversion_rules(var, da, cfg, "model", unit)
                gm = area_weighted_global_mean(da)
                agm = annual_weighted_mean(gm)
                member_agm[m] = agm
            models_agm[model_name] = member_agm

        # --- ERA5 annual global mean (with optional offsets)
        era5_da = open_era5_da(cfg, var=var, start=start, end=end, plev=plev)
        era5_da, unit = conversion_rules(var, era5_da, cfg, "era5", unit)

        agm_era5_by_offset = {}
        lrg_era5_by_offset = {}
        trend_era5_by_offset = {}

        offsets = sorted({cfg.datasets.models[m].era5_offset_k for m in plot_cfg.models})
        # always include baseline ERA5 
        if 0 not in offsets:
            offsets = [0] + offsets

        for offset_k in offsets:
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

        # --- plotting (colours from cfg)
        fig, ax = plt.subplots(figsize=tuple(plot_cfg.figsize), constrained_layout=True)

        # ERA5 (0K) solid + trend dashed
        years_era5 = agm_era5_by_offset[0].time.dt.year.values
        ax.plot(years_era5, agm_era5_by_offset[0].values, color="black", linewidth=1.5,
                label=f"ERA5 (Trend: {trend_era5_by_offset[0]} {unit}/decade)", zorder=5)
        ax.plot(years_era5, lrg_era5_by_offset[0], color="black", linewidth=1.2, linestyle="--", alpha=0.9, zorder=7)

        # extra ERA5 offset trend lines
        if plot_cfg.show_era5_offset_trends:
            for offset_k in offsets:
                if offset_k == 0:
                    continue
                ax.plot(years_era5, lrg_era5_by_offset[offset_k], color="black",
                        linewidth=1.2, linestyle="--", alpha=0.7, zorder=6)
                label_line_along_slope(ax, years_era5, lrg_era5_by_offset[offset_k], text=f"ERA5 trend +{offset_k}K", angle_boost=plot_cfg.angle, fontsize=8, alpha=0.7)

        # model members thin
        for i, model_name in enumerate(plot_cfg.models):
            c = select_colour(model_name, plot_cfg)
            for m, da in models_agm[model_name].items():
                years = da.time.dt.year.values
                ax.plot(years, da.values, color=c, alpha=0.18, linewidth=0.9, zorder=2)

        # spread + mean + mean trend
        for i, model_name in enumerate(plot_cfg.models):
            c = select_colour(model_name, plot_cfg)
            fill = select_light_colour(model_name, plot_cfg)
            ds = minmax_ds[model_name]
            years = ds.time.dt.year.values

            ax.fill_between(years, ds["min"].values, ds["max"].values,
                            color=fill, alpha=0.30,
                            label=f"{cfg.datasets.models[model_name].proper_name} (Ens. trend: {trend_mean_ens[model_name]} {unit}/decade)",
                            zorder=1)

            ax.plot(years, ds["mean"].values, color=c, linewidth=1.2, alpha=0.95, zorder=4)
            ax.plot(years, lrg_mean_ens[model_name], color=c, linewidth=1.0, linestyle="--", zorder=5)

        plev_title = ""
        plev_tag = ""
        if plev is not None:
            plev_pa = int(float(plev) * 100) if float(plev) < 2000 else int(float(plev))
            plev_hpa = int(plev_pa / 100)
            plev_title = f" at {plev_hpa} hPa"
            plev_tag = f"@{plev_hpa}hPa"

        # ax.set_title(f"{plot_cfg.title.format(var=var)}{plev_title}\n({start} – {end})", pad=10)
        if plot_cfg.title:
            title = plot_cfg.title.format(var=var, long_name=long_name)
        else:
            title = f"Global Annual Mean {long_name} ({var}{plev_title})"

        ax.set_title(f"{title}\n({start} – {end})", pad=10)
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{long_name} ({unit})")

        # major ticks every x years, minor every y year
        ax.xaxis.set_major_locator(mticker.MultipleLocator(plot_cfg.ticks.major))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(plot_cfg.ticks.minor))
        ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
        ax.grid(False, which="minor")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.01) # slight margin so traces don't touch frame

        ax.legend(loc=plot_cfg.legend.loc, frameon=False, fontsize="x-small", borderaxespad=0.0)

        if cfg.out.savefig:
            outdir = os.path.join(
                hydra.utils.get_original_cwd(),
                cfg.out.dir
            )
            os.makedirs(outdir, exist_ok=True)

            model_tags = "_".join(model_abbrev(m) for m in plot_cfg.models)
            start_tag = start.replace("-", "")
            end_tag = end.replace("-", "")
            fname = f"{var}{plev_tag}_{model_tags}_{start_tag}-{end_tag}.png"
            fig.savefig(
                os.path.join(outdir, fname),
                dpi=cfg.out.dpi,
                bbox_inches="tight"
            )
            plt.close(fig)
        else:
            plt.show()
