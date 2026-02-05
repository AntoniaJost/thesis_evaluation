# evaluation/metrics/bias_map.py
from __future__ import annotations

import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib as mpl
import cartopy.crs as ccrs
import os
import hydra
from cartopy.util import add_cyclic_point

from evaluation.general_functions import (
    ensure_allowed_var,
    resolve_period,
    open_model_da,
    open_era5_da,
    ensemble_mean_as_member,
    conversion_rules,
)


def compute_slope_per_gridpoint(da: xr.DataArray) -> xr.DataArray:
    year = da["time"].dt.year
    days_in_year = da["time"].dt.is_leap_year.astype(int) + 365
    frac = (da["time"].dt.dayofyear - 1) / days_in_year
    t = xr.DataArray((year + frac).astype("float64"), dims="time", coords={"time": da["time"]})
    da2 = da.assign_coords(t=t).swap_dims({"time": "t"}).drop_vars("time")
    fit = da2.polyfit(dim="t", deg=1, skipna=True)
    slope = fit["polyfit_coefficients"].sel(degree=1).rename("slope")
    return slope


def plot_map(ax, ds: xr.DataArray, title: str, levels, norm, cmap="RdBu_r"):
    ax.coastlines(linewidth=0.9, color="green")
    ax.set_title(title, fontsize=10)
    ax.gridlines(linewidth=0.7, color="black", alpha=0.5, linestyle="--")

    data_cyc, lon_cyc = add_cyclic_point(ds.values, coord=ds["lon"].values)
    lon = lon_cyc
    lat = ds["lat"]
    cf = ax.contourf(
        lon, lat, data_cyc,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    return cf


def run(cfg):
    plot_cfg = cfg.plots.bias_map #.metrics.bias_map
    ensure_allowed_var(cfg, plot_cfg.variable)

    start, end = resolve_period(cfg, plot_cfg)
    var = plot_cfg.variable

    # ERA5 slope
    da_era5 = conversion_rules(var, open_era5_da(cfg, var, start, end), cfg)
    era5_slope = compute_slope_per_gridpoint(da_era5) * 10.0  # per decade

    for model_name in plot_cfg.models:
        model_cfg = cfg.datasets.models[model_name]

        model_slope = {}
        bias_slope = {}

        for m in cfg.members:
            da_model = conversion_rules(
                var,
                open_model_da(model_cfg, cfg, m, var, model_cfg.modelname, plot_cfg.freq, start, end, grid=plot_cfg.grid),
                cfg
            )
            s = compute_slope_per_gridpoint(da_model) * 10.0
            model_slope[m] = s
            bias_slope[m] = (s - era5_slope).rename("bias")

        if cfg.include_ensemble_mean_as_member:
            model_slope = ensemble_mean_as_member(model_slope, name="mean")
            bias_slope = {k: (model_slope[k] - era5_slope) for k in model_slope.keys()}

        members = list(cfg.members) + (["mean"] if cfg.include_ensemble_mean_as_member else [])
        ncols = len(members)
        nrows = 3

        # colour settings from cfg
        cmap = mpl.cm.seismic
        bounds = list(plot_cfg.levels_model)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend="both")

        # build diff levels
        dmin = float(min([np.nanmin(bias_slope[m].values) for m in members]))
        dmax = float(max([np.nanmax(bias_slope[m].values) for m in members]))
        bin_size = float(plot_cfg.diff_bin)
        uL = math.floor(dmin / bin_size)
        oL = math.ceil(dmax / bin_size)
        levels_diff = [i * bin_size for i in range(uL, 0)] + [i * bin_size for i in range(1, oL + 1)]
        levels_diff = sorted([x for x in levels_diff if x != 0])

        cmap_diff = cm.BrBG
        norm_diff = mcolors.BoundaryNorm(boundaries=levels_diff, ncolors=cmap_diff.N)

        fig, axes = plt.subplots(
            ncols=ncols,
            nrows=nrows,
            figsize=(plot_cfg.figscale_col * ncols, plot_cfg.figscale_row * nrows),
            layout="constrained",
            subplot_kw=dict(projection=ccrs.Robinson()),
            squeeze=False,
        )
        fig.suptitle(plot_cfg.title.format(var=var, model=model_name), fontsize=15, fontweight="bold")

        for r in range(nrows):
            for c in range(ncols):
                mem = members[c]
                ax = axes[r, c]
                if r == 0:
                    cf = plot_map(ax, model_slope[mem], f"{model_name}\n{mem}", levels=bounds, norm=norm, cmap=cmap)
                elif r == 1:
                    cf = plot_map(ax, era5_slope, "ERA5", levels=bounds, norm=norm, cmap=cmap)
                else:
                    cf = plot_map(ax, bias_slope[mem], f"Diff\n{mem} - ERA5",
                                  levels=levels_diff, norm=mcolors.CenteredNorm(), cmap=cmap_diff)

        # colourbars
        cbar1 = fig.colorbar(cf, ax=axes[0:2, :], orientation="vertical",
                             shrink=0.7, pad=0.02, spacing="proportional")
        cbar1.set_label(plot_cfg.cbar_label_model)

        cbar2 = fig.colorbar(axes[2, 0].collections[0], ax=axes[2, :], orientation="vertical",
                             shrink=0.7, pad=0.02, spacing="proportional")
        cbar2.set_label(plot_cfg.cbar_label_diff)

        # plt.show()
        if cfg.out.savefig:
            outdir = os.path.join(
                hydra.utils.get_original_cwd(),
                cfg.out.dir
            )
            os.makedirs(outdir, exist_ok=True)

            fname = "bias_map.png"
            fig.savefig(
                os.path.join(outdir, fname),
                dpi=cfg.out.dpi,
                bbox_inches="tight"
            )
            plt.close(fig)
        else:
            plt.show()