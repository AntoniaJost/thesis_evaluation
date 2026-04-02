from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import hydra
import warnings

from evaluation.general_functions import (
    model_abbrev,
    open_model_da,
    open_era5_da,
    ensemble_mean_as_member,
    conversion_rules,
    should_compute_output,
    iter_vars_and_plevs,
    plev_strings,
    format_unit_for_plot
)

from evaluation.metrics.bias_map import (
    plot_map,
    add_row_labels,
    add_bottom_numbers,
    area_weighted_mean_map,
    area_weighted_rmse_map,
)

from evaluation.metrics.individual_plots import (
    _prepare_field, # this one contains the detrending logic
    _selection_bounds_for_freq,
    _get_map_bounds,
    _time_stat,
    _map_levels_and_ticks,
    _get_map_norm
)


def _validate_cfg(plot_cfg):
    stat = _time_stat(plot_cfg)
    if stat not in {"raw"}:
        raise ValueError(
            f"plots.diff_map_raw.time_stat must be 'raw', got: {stat}"
        )


def run(cfg):
    plot_cfg = cfg.plots.diff_map_raw
    _validate_cfg(plot_cfg)
    if plot_cfg.detrend.enabled and not plot_cfg.detrend.preserve_mean:
        warnings.warn(
            "You have chosen to plot a detrended map, without readding the mean. This will lead to very small values very close to zero. Therefore this plotting takes very long (approx. 30 min. per 1 model, 1 variable, 1 pressure level)."
        )
    add_dir = str(plot_cfg.special_outdir) if plot_cfg.special_outdir else ""
    centre = float(getattr(plot_cfg, "global_centre", 0))

    for item in iter_vars_and_plevs(cfg, plot_cfg):
        var = item["var"]
        long_name = item["long_name"]
        unit = format_unit_for_plot(item["unit"])
        start = item["start"]
        end = item["end"]

        start_sel, end_sel = _selection_bounds_for_freq(start, end, plot_cfg.freq)
        single_time = pd.Timestamp(start) == pd.Timestamp(end)

        for plev in item["plevs"]:
            plev_title, plev_tag = plev_strings(plev)

            for model_name in plot_cfg.models:
                proper_model_name = cfg.datasets.models[model_name].proper_name
                model_cfg = cfg.datasets.models[model_name]

                outdir = os.path.join(
                    hydra.utils.get_original_cwd(),
                    cfg.out.dir,
                    "diff_map_raw",
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
                stat_tag = "raw" if _time_stat(plot_cfg) == "raw" else ""

                perc = ""
                if plot_cfg.range_source.percentile == 99:
                    perc = "_99p"
                elif plot_cfg.range_source.percentile == 95:
                    perc = "_95p"

                if plot_cfg.season == None or plot_cfg.season == "full":
                    season_tag = ""
                else:
                    season_tag = f"_{plot_cfg.season}"
                detrend_tag = "_detrended" if plot_cfg.detrend.enabled else ""
                fname = f"diff_raw_{var}{plev_tag}_{model_tag}{season_tag}_{stat_tag}_{start_tag}-{end_tag}{detrend_tag}{perc}.png"
                outfile = os.path.join(outdir, fname)

                if cfg.out.savefig and not should_compute_output(outfile, getattr(cfg.out, "overwrite", "ask")):
                    continue

                unit_here = unit

                # ERA5 prepared field
                da_era5 = open_era5_da(cfg, var=var, start=start_sel, end=end_sel, plev=plev, freq=plot_cfg.freq, grid=plot_cfg.grid)
                era5_source = "era5_cmor" if plot_cfg.freq == "daily" else "era5_natural"
                da_era5, unit_here = conversion_rules(var, da_era5, cfg, era5_source, unit_here)
                era5_field = _prepare_field(da_era5, plot_cfg, method="map", start=start_sel, end=end_sel, season=plot_cfg.season)

                model_field = {}
                diff_field = {}

                # model prepared fields
                for m in cfg.members:
                    da_model = open_model_da(
                        model_cfg=model_cfg,
                        cfg=cfg,
                        member=m,
                        var=var,
                        modelname=model_cfg.modelname,
                        freq=plot_cfg.freq,
                        start=start_sel,
                        end=end_sel,
                        grid=plot_cfg.grid,
                        plev=plev,
                    )
                    da_model, _ = conversion_rules(var, da_model, cfg, "model", unit_here)
                    prepared = _prepare_field(da_model, plot_cfg, method="map", start=start_sel, end=end_sel, season=plot_cfg.season)
                    model_field[m] = prepared
                    diff_field[m] = (prepared - era5_field).rename("difference")

                if plot_cfg.include_ensemble_mean_as_member:
                    model_field = ensemble_mean_as_member(model_field, name="mean")
                    diff_field = {k: (model_field[k] - era5_field) for k in model_field.keys()}

                # statistics
                diff_stats = {}
                rmse_stats = {}
                if plot_cfg.add_numbers:
                    for mem_name, diff_map in diff_field.items():
                        diff_stats[mem_name] = area_weighted_mean_map(diff_map)
                        rmse_stats[mem_name] = area_weighted_rmse_map(diff_map)

                members = list(cfg.members) + (["mean"] if plot_cfg.include_ensemble_mean_as_member else [])
                ncols = len(members)
                nrows = 3

                # colour settings for model / ERA5 rows
                cmap_model = mpl.cm.get_cmap(plot_cfg.colourbar.cmap_model)
                if plot_cfg.colourbar.manual_vmin and plot_cfg.colourbar.manual_vmax:
                    vmin_model = float(plot_cfg.colourbar.manual_vmin)
                    vmax_model = float(plot_cfg.colourbar.manual_vmax)
                else:
                    vmin_model, vmax_model = _get_map_bounds(
                        cfg=cfg,
                        plot_cfg=plot_cfg,
                        arrays=list(model_field.values()) + [era5_field],
                        var=var,
                        plev=plev,
                        difference=False,
                        anomaly=False,
                        single_time=single_time,
                )
                if plot_cfg.colourbar.use_custom_bins:
                    levels_model, ticks_model = _map_levels_and_ticks(vmin_model, vmax_model, plot_cfg.colourbar.ticks_everyX_model, plot_cfg.colourbar.keep_0_tick_model, plot_cfg)
                else:
                    levels_model = np.linspace(vmin_model, vmax_model, 21)
                    ticks_model = None
                norm_model = _get_map_norm(plot_cfg, vmin_model, vmax_model)

                # colour settings for difference row
                cmap_diff = mpl.cm.get_cmap(plot_cfg.colourbar.cmap_diff)
                vmin_diff, vmax_diff = _get_map_bounds(
                    cfg=cfg,
                    plot_cfg=plot_cfg,
                    arrays=list(diff_field.values()),
                    var=var,
                    plev=plev,
                    difference=True,
                    anomaly=False,
                    single_time=single_time,
                )
                if plot_cfg.colourbar.use_custom_bins:
                    levels_diff, ticks_diff = _map_levels_and_ticks(vmin_diff, vmax_diff,  plot_cfg.colourbar.ticks_everyX_diff, plot_cfg.colourbar.keep_0_tick_diff, plot_cfg)
                else:
                    levels_diff = np.linspace(vmin_diff, vmax_diff, 21)
                    ticks_diff = None
                norm_diff = _get_map_norm(plot_cfg, vmin_diff, vmax_diff)

                coastline_colour = str(plot_cfg.coastline_colour)

                # plotting
                fig, axes = plt.subplots(
                    ncols=ncols,
                    nrows=nrows,
                    figsize=(plot_cfg.figscale_col * ncols, plot_cfg.figscale_row * nrows),
                    layout="constrained",
                    subplot_kw=dict(projection=ccrs.Robinson(central_longitude=centre)),
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
                        time_stat=_time_stat(plot_cfg),
                        start=start_str,
                        end=end_str,
                        season=plot_cfg.season,
                    )
                else:
                    # stat_label = "raw field" if _time_stat(plot_cfg) == "raw" else ""
                    if plot_cfg.season == None or plot_cfg.season == "full":
                        title = f"{long_name} ({var}{plev_title}): {proper_model_name} vs ERA5 | {start_str} to {end_str}"
                    else:
                        title = f"{long_name} ({var}{plev_title}): {proper_model_name} vs ERA5 | {plot_cfg.season} | {start_str} to {end_str}"

                if plot_cfg.detrend.enabled and plot_cfg.detrend.preserve_mean:
                    title += " (detrended, mean readded)"
                elif plot_cfg.detrend.enabled and not plot_cfg.detrend.preserve_mean:
                    title += " (detrended)"

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
                                model_field[mem],
                                mem,
                                levels=levels_model,
                                norm=norm_model,
                                cmap=cmap_model,
                                coastline_colour=coastline_colour,
                            )
                        elif r == 1:
                            cf_model = plot_map(
                                ax,
                                era5_field,
                                "",
                                levels=levels_model,
                                norm=norm_model,
                                cmap=cmap_model,
                                coastline_colour=coastline_colour,
                            )
                        else:
                            cf_diff = plot_map(
                                ax,
                                diff_field[mem],
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
                                    unit_here,
                                )

                add_row_labels(axes, [proper_model_name, "ERA5", "Difference"])

                cbar1 = fig.colorbar(
                    cf_model,
                    ax=axes[0:2, :],
                    orientation="vertical",
                    shrink=0.7,
                    pad=0.02,
                    spacing="proportional",
                )
                if plot_cfg.cbar_label_model:
                    cbar1.set_label(f"{plot_cfg.cbar_label_model} ({unit_here})")
                else:
                    cbar1.set_label(f"{var} ({unit_here})")
                cbar1.set_ticks(ticks_model)

                cbar2 = fig.colorbar(
                    cf_diff,
                    ax=axes[2, :],
                    orientation="vertical",
                    shrink=0.7,
                    pad=0.02,
                    spacing="proportional",
                )
                cbar2.set_label(f"{plot_cfg.cbar_label_diff} ({unit_here})")
                cbar2.set_ticks(ticks_diff)

                if cfg.out.savefig:
                    fig.savefig(
                        outfile,
                        dpi=cfg.out.dpi,
                        bbox_inches="tight",
                    )
                    plt.close(fig)
                else:
                    plt.show()