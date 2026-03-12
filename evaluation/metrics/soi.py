# evaluation/metrics/soi.py
from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerBase
import os
import hydra

from evaluation.general_functions import (
    ensure_allowed_var,
    resolve_period,
    open_model_da,
    open_era5_da,
    ensemble_mean_as_member,
    conversion_rules,
    should_compute_output,
    model_abbrev,
)


def _lat_slice(da, lat0, lat1):
    lat = da["lat"]
    # if ascending, slice(low, high); if descending, slice(high, low)
    if lat[0] < lat[-1]:
        # slice(-20, -10, None)
        return slice(min(lat0, lat1), max(lat0, lat1))
    else:
        # slice(-10, -20, None)
        return slice(max(lat0, lat1), min(lat0, lat1))


def sSLP(x, N: int):
    std = np.sqrt(np.sum((x - x.mean())**2) / N)
    sSLP = (x - x.mean()) / std
    return sSLP


def calc_soi(ds: xr.Dataset, start=None, end=None, slp_var: str = "psl",
             tahiti_box=None, darwin_box=None):
    slp = ds[slp_var] / 100.0  # Pa -> hPa

    lat_slice = _lat_slice(slp, tahiti_box["lat0"], tahiti_box["lat1"])
    tahiti = slp.sel({"lat": lat_slice, "lon": slice(tahiti_box["lon0"], tahiti_box["lon1"])})
    darwin = slp.sel({"lat": lat_slice, "lon": slice(darwin_box["lon0"], darwin_box["lon1"])})
    tahiti_mean = tahiti.mean(("lat", "lon"))
    darwin_mean = darwin.mean(("lat", "lon"))

    # time frame
    if start is None:
        start = str(tahiti_mean.time.values[0])[:7] # first available date
    if end is None:
        end = str(tahiti_mean.time.values[-1])[:7]
    clim_tahiti = tahiti_mean.sel(time=slice(start, end))
    clim_darwin = darwin_mean.sel(time=slice(start, end))

    N = len(clim_tahiti.time.values)
    s_t = sSLP(clim_tahiti, N)
    s_d = sSLP(clim_darwin, N)

    std_monthly = np.sqrt(np.sum((s_t - s_d) ** 2) / N)
    soi = (s_t - s_d) / std_monthly
    return soi

# function for halfed bar in legend is written by ChatGPT
class SplitBarHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        left = mpatches.Rectangle((xdescent, ydescent), width / 2, height,
                                  facecolor="steelblue", alpha=0.6, transform=trans)
        right = mpatches.Rectangle((xdescent + width / 2, ydescent), width / 2, height,
                                   facecolor="indianred", alpha=0.6, transform=trans)
        return [left, right]

# function for colour block with line inside has also been written by ChatGPT
class HistKDEHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height,
                       fontsize, trans):

        hist_color, line_color = orig_handle

        # histogram block
        rect = mpatches.Rectangle(
            (xdescent, ydescent),
            width, height,
            facecolor=hist_color,
            edgecolor="none",
            alpha=0.6,
            transform=trans
        )

        # KDE line through the block
        line = mlines.Line2D(
            [xdescent, xdescent + width],
            [ydescent + height / 2, ydescent + height / 2],
            color=line_color,
            linewidth=2,
            transform=trans
        )

        return [rect, line]
    
    
def build_members_to_plot(cfg, plot_cfg, soi_members: dict[str, xr.DataArray]) -> list[str]:
    """
    decide which member plots to produce
    rules:
    - only_mean=True  -> only ['mean']
    - only_mean=False -> all real members from cfg.members
      and, if include_ensemble_mean_as_member=True, append 'mean'
    """
    if plot_cfg.only_mean:
        if "mean" not in soi_members:
            raise ValueError(
                "plots.soi.only_mean=true requires "
                "plots.soi.include_ensemble_mean_as_member=true."
            )
        return ["mean"]

    members = list(cfg.members)
    if plot_cfg.include_ensemble_mean_as_member and "mean" in soi_members:
        members.append("mean")
    return members


def soi_output_filename(model_name: str, member: str, start: str, end: str, hist_enabled: bool) -> str:
    model_tag = model_abbrev(model_name)
    start_tag = str(start)[:10].replace("-", "")
    end_tag = str(end)[:10].replace("-", "")
    suffix = "hist_kde" if hist_enabled else "ts"
    return f"soi_{suffix}_{model_tag}_{member}_{start_tag}-{end_tag}.png"


def run(cfg):
    plot_cfg = cfg.plots.soi
    ensure_allowed_var(cfg, plot_cfg.variable)

    if plot_cfg.variable != "psl":
        raise ValueError("SOI only supports variable='psl'.")

    start, end = resolve_period(cfg, plot_cfg)

    outdir = os.path.join(
        hydra.utils.get_original_cwd(),
        cfg.out.dir,
        "soi",
    )
    os.makedirs(outdir, exist_ok=True)

    # --- load ERA5
    da_era5 = open_era5_da(cfg, var=plot_cfg.variable, start=start, end=end)
    da_era5, _ = conversion_rules(plot_cfg.variable, da_era5, cfg, "era5")

    era5_var_name = cfg.variables.era5_name[plot_cfg.variable]
    era5_ds = da_era5.to_dataset(name=era5_var_name)

    soi_era5 = calc_soi(
        era5_ds,
        start=start[:7],
        end=end[:7],
        slp_var=era5_var_name,
        tahiti_box=plot_cfg.regions.tahiti,
        darwin_box=plot_cfg.regions.darwin,
    ).to_pandas().dropna()

    # --- loop over requested models
    for model_name in plot_cfg.models:
        model_cfg = cfg.datasets.models[model_name]
        proper_model_name = getattr(model_cfg, "proper_name", model_name)

        # compute SOI for all configured real members first
        soi_members = {}
        for member in cfg.members:
            da_model = open_model_da(
                model_cfg=model_cfg,
                cfg=cfg,
                member=member,
                var=plot_cfg.variable,
                modelname=model_cfg.modelname,
                freq=plot_cfg.freq,
                start=start,
                end=end,
                grid=plot_cfg.grid,
            )
            da_model, _ = conversion_rules(plot_cfg.variable, da_model, cfg, "model")

            ds_model = da_model.to_dataset(name=plot_cfg.variable)

            soi_members[member] = calc_soi(
                ds_model,
                start=start[:7],
                end=end[:7],
                slp_var=plot_cfg.variable,
                tahiti_box=plot_cfg.regions.tahiti,
                darwin_box=plot_cfg.regions.darwin,
            )

        # optionally add ensemble mean as pseudo-member
        if plot_cfg.include_ensemble_mean_as_member:
            soi_members = ensemble_mean_as_member(soi_members, name="mean")

        members_to_plot = build_members_to_plot(cfg, plot_cfg, soi_members)

        for mem in members_to_plot:
            model_series = soi_members[mem].to_pandas().dropna()

            fname = soi_output_filename(
                model_name=model_name,
                member=mem,
                start=start,
                end=end,
                hist_enabled=plot_cfg.hist_kde.enabled,
            )
            outfile = os.path.join(outdir, fname)

            if cfg.out.savefig and not should_compute_output(
                outfile, getattr(cfg.out, "overwrite", "ask")
            ):
                continue

            # --- plotting
            fig = plt.figure(figsize=tuple(plot_cfg.figsize))
            gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)

            ax_ts = fig.add_subplot(gs[0, 0])
            ax_hist = fig.add_subplot(gs[0, 1], sharey=ax_ts) if plot_cfg.hist_kde.enabled else None

            # --- LEFT: ENSO / SOI time series
            colours_model = np.where(
                model_series >= 0,
                plot_cfg.enso_plot.colour_elnino,
                plot_cfg.enso_plot.colour_lanina,
            )

            ax_ts.bar(
                model_series.index,
                model_series.values,
                width=plot_cfg.bar_width_days,
                color=colours_model,
                alpha=0.6,
                zorder=1,
            )

            ax_ts.plot(
                soi_era5.index,
                soi_era5.values,
                color="black",
                linewidth=1.2,
                zorder=2,
            )

            ax_ts.axhline(0, color="black", linewidth=1)
            ax_ts.set_xlabel("Time")
            ax_ts.set_ylabel("Southern Oscillation Index")
            ax_ts.set_title(
                f"SOI: ERA5 vs {proper_model_name} ({mem})",
                fontsize=14,
                weight="bold",
            )
            ax_ts.grid(True, linestyle="--", alpha=0.3)

            era5_handle = plt.Line2D([], [], color="black", linewidth=1.8, label="ERA5")
            model_handle = mpatches.Rectangle((0, 0), 1, 1, label=proper_model_name)

            ax_ts.legend(
                handles=[era5_handle, model_handle],
                handler_map={model_handle: SplitBarHandler()},
                frameon=True,
                loc="upper left",
            )

            # --- RIGHT: histogram + KDE
            if plot_cfg.hist_kde.enabled:
                bins = plot_cfg.hist_kde.hist_bins
                hist_pad = plot_cfg.hist_kde.hist_pad
                kde_n_eval = plot_cfg.hist_kde.kde_n_eval
                bw_method = str(plot_cfg.hist_kde.kde_bw_method)
                xpad = plot_cfg.hist_kde.xpad

                modelcol = str(plot_cfg.hist_kde.model_colour)
                model_linecol = str(plot_cfg.hist_kde.model_linecol)
                era5col = str(plot_cfg.hist_kde.era5_colour)
                era5_linecol = str(plot_cfg.hist_kde.era5_linecol)

                ax_hist.hist(
                    soi_era5.values,
                    bins=bins,
                    density=True,
                    histtype="stepfilled",
                    color=era5col,
                    alpha=0.6,
                    orientation="horizontal",
                )

                ax_hist.hist(
                    model_series.values,
                    bins=bins,
                    density=True,
                    histtype="stepfilled",
                    color=modelcol,
                    alpha=0.3,
                    orientation="horizontal",
                )

                ymin = min(soi_era5.min(), model_series.min())
                ymax = max(soi_era5.max(), model_series.max())
                y = np.linspace(ymin - hist_pad, ymax + hist_pad, kde_n_eval)

                kde_era5 = gaussian_kde(soi_era5.values, bw_method=bw_method)
                kde_model = gaussian_kde(model_series.values, bw_method=bw_method)

                ax_hist.plot(kde_era5(y), y, color=era5_linecol, lw=2)
                ax_hist.plot(kde_model(y), y, color=model_linecol, lw=2)

                ax_hist.set_xlim(-xpad, None)
                ax_hist.axhline(0, color="black", linewidth=1)
                ax_hist.set_xlabel("Density")
                ax_hist.grid(True, linestyle="--", alpha=0.3)
                ax_hist.set_title("Probability Distribution", fontsize=10, weight="bold")

                plt.setp(ax_hist.get_yticklabels(), visible=False)
                ax_hist.tick_params(axis="y", length=0)

                era5_handle_hist = (era5col, era5_linecol)
                model_handle_hist = (modelcol, model_linecol)

                ax_hist.legend(
                    handles=[era5_handle_hist, model_handle_hist],
                    labels=["ERA5", proper_model_name], # or better just "Model" (shorter)?
                    handler_map={tuple: HistKDEHandler()},
                    frameon=True,
                )

            if cfg.out.savefig:
                fig.savefig(
                    outfile,
                    dpi=cfg.out.dpi,
                    bbox_inches="tight",
                )
                plt.close(fig)
            else:
                plt.show()
