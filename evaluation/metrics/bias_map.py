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
    model_abbrev,
    open_model_da,
    open_era5_da,
    ensemble_mean_as_member,
    conversion_rules,
    should_compute_output,
    iter_vars_and_plevs,
    plev_strings,
    get_range_from_csv,
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


def plot_map(ax, ds: xr.DataArray, title: str, levels, norm, cmap="RdBu_r", coastline_colour: str = "black"):
    ax.coastlines(linewidth=0.9, color=coastline_colour)
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


def add_row_labels(axes, labels, x=-0.05, fontsize=10, fontweight="bold"):
    # add a label for each row on the left side
    for r, label in enumerate(labels):
        ax = axes[r, 0]
        ax.text(
            x,
            0.5,
            label,
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=fontsize,
            fontweight=fontweight,
        )


def area_weights_2d(da: xr.DataArray, lat_name: str = "lat") -> xr.DataArray:
    return np.cos(np.deg2rad(da[lat_name])) # latitude weights for area-weighted map statistics


def area_weighted_mean_map(da: xr.DataArray, lat_name: str = "lat", lon_name: str = "lon") -> float:
    w = area_weights_2d(da, lat_name=lat_name)
    return float(da.weighted(w).mean(dim=(lat_name, lon_name)).values)


def area_weighted_rmse_map(diff: xr.DataArray, lat_name: str = "lat", lon_name: str = "lon") -> float:
    w = area_weights_2d(diff, lat_name=lat_name)
    mse = (diff ** 2).weighted(w).mean(dim=(lat_name, lon_name))
    return float(np.sqrt(mse.values))


def add_bottom_numbers(ax, diff_value: float, rmse_value: float, unit: str, fontsize: int = 8):
    # add difference and RMSE as number below the panel
    text = f"Diff: {diff_value:+.2f} | RMSE: {rmse_value:.2f} {unit}"
    ax.text(
        0.5,
        -0.14,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        clip_on=False,
    )


def get_slope_range_from_csv(cfg, csv_file: str, var: str, plev: int | None):
    percentile = cfg.plots.bias_map.range_source.percentile
    return get_range_from_csv(
        percentile=percentile,
        csv_file=csv_file,
        var=var,
        plev=plev,
        prefix="slope"
    )


def nice_bin_size(vmin: float, vmax: float, target_bins: int = 12):
    """
    compute a "nice" bin size (fractions of 1, 2 or 5; nothing like 1.293)
    target_bins ≈ desired number of bins across the full range
    """
    span = abs(vmax - vmin)
    raw_step = span / target_bins
    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / (10 ** exponent)

    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    step = nice_fraction * 10 ** exponent
    return step


def build_zero_bin_levels(vmin: float, vmax: float, bin_size: float):
    """
    build discrete levels with a white bin around zero
    """
    lower = math.floor(vmin / bin_size)
    upper = math.ceil(vmax / bin_size)

    levels = [i * bin_size for i in range(lower, 0)] + [i * bin_size for i in range(1, upper + 1)]
    levels = sorted(set(levels))

    # make sure the central white interval exists
    if -bin_size not in levels:
        levels.append(-bin_size)
    if bin_size not in levels:
        levels.append(bin_size)

    levels = sorted(levels)
    return levels


def symmetric_ticks_from_levels(levels, vmin, vmax, keep_every: int = 1, include_zero: bool = True):
    """
    bit of a work around function to make sure that the ticks span symmatrically around
    the white bin -> uses the first positive boundary outside the zero bin as the tick spacing
    keep_every means keep every nth tick away from zero, 1 means keep all
    include_zero means whether to include 0 as a tick
    """
    levels = np.array(sorted(levels), dtype=float)

    # determine tick spacing from first positive level outside white bin
    pos_levels = levels[levels > 0]
    if len(pos_levels) == 0:
        return [0]

    step = pos_levels[0]

    # get distances from zero
    max_pos_n = int(np.ceil(vmax / step))
    max_neg_n = int(np.ceil(abs(vmin) / step))

    if keep_every < 1:
        raise ValueError(f"keep_every must be >= 1, got {keep_every}")

    # count ticks by distance from zero to ensure symmetry
    pos_n = np.arange(1, max_pos_n + 1)
    neg_n = np.arange(1, max_neg_n + 1)

    # keep every nth tick, starting with n = keep_every
    pos_n = pos_n[pos_n % keep_every == 0]
    neg_n = neg_n[neg_n % keep_every == 0]

    pos_ticks = pos_n * step
    neg_ticks = -neg_n * step

    ticks = np.concatenate([neg_ticks[::-1], ([0] if include_zero else []), pos_ticks])
    return ticks


def run(cfg):
    plot_cfg = cfg.plots.bias_map 
    add_dir = str(plot_cfg.special_outdir) if plot_cfg.special_outdir else ""

    for item in iter_vars_and_plevs(cfg, plot_cfg):
        var = item["var"]
        long_name = item["long_name"]
        unit = item["unit"]
        start = item["start"]
        end = item["end"]

        for plev in item["plevs"]:
            plev_title, plev_tag = plev_strings(plev)

            for model_name in plot_cfg.models:
                proper_model_name = cfg.datasets.models[model_name].proper_name
                model_cfg = cfg.datasets.models[model_name]

                outdir = os.path.join(
                    hydra.utils.get_original_cwd(),
                    cfg.out.dir,
                    "bias_map",
                    add_dir,
                )
                os.makedirs(outdir, exist_ok=True)

                model_tag = model_abbrev(model_name)
                if plot_cfg.freq == "monthly":
                    start_tag = start[:7].replace("-", "")
                    end_tag = end[:7].replace("-", "")
                else:
                    start_tag = start.replace("-", "")
                    end_tag = end.replace("-", "")
                perc = ""
                if plot_cfg.range_source.percentile == 99:
                    perc = "_99p"
                elif plot_cfg.range_source.percentile == 95:
                    perc = "_95p"
                fname = f"bias_map_{var}{plev_tag}_{model_tag}_{start_tag}-{end_tag}{perc}.png"
                outfile = os.path.join(outdir, fname)

                if cfg.out.savefig and not should_compute_output(outfile, getattr(cfg.out, "overwrite", "ask")):
                    continue

                unit_here = unit

                # ERA5 slope
                da_era5 = open_era5_da(cfg, var=var, start=start, end=end, plev=plev)
                da_era5, unit_here = conversion_rules(var, da_era5, cfg, "era5", unit_here)
                era5_slope = compute_slope_per_gridpoint(da_era5) * 10.0  # per decade

                model_slope = {}
                bias_slope = {}

                # model slope
                for m in cfg.members:
                    da_model = open_model_da(model_cfg=model_cfg, cfg=cfg, member=m, var=var, modelname=model_cfg.modelname, freq=plot_cfg.freq, start=start, end=end, grid=plot_cfg.grid, plev=plev,)
                    da_model, _ = conversion_rules(var, da_model, cfg, "model", unit_here)
                    s = compute_slope_per_gridpoint(da_model) * 10.0
                    model_slope[m] = s
                    bias_slope[m] = (s - era5_slope).rename("bias")

                if plot_cfg.include_ensemble_mean_as_member:
                    model_slope = ensemble_mean_as_member(model_slope, name="mean")
                    bias_slope = {k: (model_slope[k] - era5_slope) for k in model_slope.keys()}

                # statistics
                diff_stats = {}
                rmse_stats = {}
                if plot_cfg.add_numbers:
                    for mem_name, diff_map in bias_slope.items():
                        diff_stats[mem_name] = area_weighted_mean_map(diff_map)
                        rmse_stats[mem_name] = area_weighted_rmse_map(diff_map)

                members = list(cfg.members) + (["mean"] if plot_cfg.include_ensemble_mean_as_member else [])
                ncols = len(members)
                nrows = 3

                # ---- colour settings from cfg, ranges from CSV; for model 
                cmap_model = mpl.cm.get_cmap(plot_cfg.cmap_model)
                csv_file_model = os.path.join(hydra.utils.get_original_cwd(), cfg.plots.bias_map.range_source.csv_file1)
                vmin_model, vmax_model = get_slope_range_from_csv(cfg, csv_file_model, var, plev)

                # automatic bin size unless manually provided
                if plot_cfg.set_size_of_bins is None:
                    bin_size = nice_bin_size(vmin_model, vmax_model, plot_cfg.target_bins)
                else:
                    bin_size = plot_cfg.set_size_of_bins

                levels_model = build_zero_bin_levels(
                    vmin=vmin_model,
                    vmax=vmax_model,
                    bin_size=bin_size,
                )
                norm_model = mpl.colors.CenteredNorm(vcenter=0) 
                ticks_model = symmetric_ticks_from_levels(levels_model, vmin_model, vmax_model, plot_cfg.ticks_everyX_model, plot_cfg.keep_0_tick_model)
                coastline_colour = str(plot_cfg.coastline_colour)
                
                # ---- colours for difference row
                cmap_diff = mpl.cm.get_cmap(plot_cfg.cmap_diff)
                csv_file_diff = os.path.join(hydra.utils.get_original_cwd(), cfg.plots.bias_map.range_source.csv_file2)
                vmin_diff, vmax_diff = get_slope_range_from_csv(cfg, csv_file_diff, var, plev)

                # automatic bin size unless manually provided
                if plot_cfg.set_size_of_bins_diff is None:
                    bin_size_diff = nice_bin_size(vmin_diff, vmax_diff, plot_cfg.target_bins_diff)
                else:
                    bin_size_diff = plot_cfg.set_size_of_bins_diff

                levels_diff = build_zero_bin_levels(
                    vmin=vmin_diff,
                    vmax=vmax_diff,
                    bin_size=bin_size_diff,
                )
                norm_diff = mpl.colors.CenteredNorm(vcenter=0) 
                ticks_diff = symmetric_ticks_from_levels(levels_diff, vmin_diff, vmax_diff, plot_cfg.ticks_everyX_diff, plot_cfg.keep_0_tick_diff)

                # ---- plotting ----
                fig, axes = plt.subplots(
                    ncols=ncols,
                    nrows=nrows,
                    figsize=(plot_cfg.figscale_col * ncols, plot_cfg.figscale_row * nrows),
                    layout="constrained",
                    subplot_kw=dict(projection=ccrs.Robinson()),
                    squeeze=False,
                )
                start_str = start[:7]  # YYYY-MM
                end_str = end[:7]
                if plot_cfg.title:
                    title = plot_cfg.title.format(
                        var=var,
                        long_name=long_name,
                        model=model_name,
                        proper_model_name=proper_model_name,
                    )
                else:
                    title = f"{long_name} ({var}{plev_title}) trend slope: {proper_model_name} vs ERA5 | {start_str} to {end_str}"

                fig.suptitle(title, fontsize=15, fontweight="bold")

                cf_model = None
                cf_diff = None

                for r in range(nrows):
                    for c in range(ncols):
                        mem = members[c]
                        ax = axes[r, c]
                        if r == 0:
                            cf_model = plot_map(
                                ax,
                                model_slope[mem],
                                mem,
                                levels=levels_model,
                                norm=norm_model,
                                cmap=cmap_model,
                                coastline_colour=coastline_colour,
                            )
                        elif r == 1:
                            cf_model = plot_map(
                                ax,
                                era5_slope,
                                "",
                                levels=levels_model,
                                norm=norm_model,
                                cmap=cmap_model,
                                coastline_colour=coastline_colour,
                            )
                        else:
                            cf_diff = 
                            (
                                ax,
                                bias_slope[mem],
                                f"{mem} - ERA5",
                                levels=levels_diff,
                                norm=norm_diff,
                                cmap=cmap_diff,
                                coastline_colour=coastline_colour,
                            )
                            if plot_cfg.add_numbers:
                                add_bottom_numbers(
                                    ax,
                                    diff_stats[mem],
                                    rmse_stats[mem],
                                    f"{unit_here}/dec",
                                )
                add_row_labels(axes, [proper_model_name, "ERA5", "Difference"])
                # colourbars
                cbar1 = fig.colorbar(cf_model, ax=axes[0:2, :], orientation="vertical",
                                    shrink=0.7, pad=0.02, spacing="proportional")
                cbar1.set_label(f"{plot_cfg.cbar_label_model} ({unit_here}/decade)")
                cbar1.set_ticks(ticks_model)

                cbar2 = fig.colorbar(cf_diff, ax=axes[2, :], orientation="vertical",
                                    shrink=0.7, pad=0.02, spacing="proportional")
                cbar2.set_label(f"{plot_cfg.cbar_label_diff} ({unit_here}/decade)")
                cbar2.set_ticks(ticks_diff)

                # plt.show()
                if cfg.out.savefig:
                    fig.savefig(
                        outfile,
                        dpi=cfg.out.dpi,
                        bbox_inches="tight"
                    )
                    plt.close(fig)
                else:
                    plt.show()