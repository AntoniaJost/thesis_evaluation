# evaluation/metrics/soi.py
from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
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
)


def _lat_slice(da: xr.DataArray, lat0: float, lat1: float):
    lat = da["lat"]
    if lat[0] < lat[-1]:
        return slice(min(lat0, lat1), max(lat0, lat1))
    else:
        return slice(max(lat0, lat1), min(lat0, lat1))


def sSLP(x, N: int):
    std = np.sqrt(np.sum((x - x.mean()) ** 2) / N)
    return (x - x.mean()) / std


def calc_soi(ds: xr.Dataset, start=None, end=None, slp_var: str = "psl",
             tahiti_box=None, darwin_box=None):
    slp = ds[slp_var] / 100.0  # Pa -> hPa

    lat_slice = _lat_slice(slp, tahiti_box["lat0"], tahiti_box["lat1"])
    tahiti = slp.sel({"lat": lat_slice, "lon": slice(tahiti_box["lon0"], tahiti_box["lon1"])})
    darwin = slp.sel({"lat": lat_slice, "lon": slice(darwin_box["lon0"], darwin_box["lat1"])})  # FIX BELOW
    # ^^^ NOTE: your original boxes used same lat slice for both; lon slice differs.
    # This line above has a typo (lon1 wrongly uses lat1). Corrected right after.

    darwin = slp.sel({"lat": lat_slice, "lon": slice(darwin_box["lon0"], darwin_box["lon1"])})

    tahiti_mean = tahiti.mean(("lat", "lon"))
    darwin_mean = darwin.mean(("lat", "lon"))

    if start is None:
        start = str(tahiti_mean.time.values[0])[:7]
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


class SplitBarHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        left = mpatches.Rectangle((xdescent, ydescent), width / 2, height,
                                  facecolor="steelblue", alpha=0.6, transform=trans)
        right = mpatches.Rectangle((xdescent + width / 2, ydescent), width / 2, height,
                                   facecolor="indianred", alpha=0.6, transform=trans)
        return [left, right]


def run(cfg):
    plot_cfg = cfg.plots.soi
    ensure_allowed_var(cfg, plot_cfg.variable)

    start, end = resolve_period(cfg, plot_cfg)

    # --- load ERA5
    da_era5 = open_era5_da(cfg, var=plot_cfg.variable, start=start, end=end)
    da_era5 = conversion_rules(plot_cfg.variable, da_era5, cfg)
    era5_ds = da_era5.to_dataset(name=cfg.variables.era5_name[plot_cfg.variable])  # keep var name accessible
    era5_slp_var = plot_cfg.get("era5_var_override", None) or cfg.variables.era5_name[plot_cfg.variable]
    soi_era5 = calc_soi(
        era5_ds, start=start[:7], end=end[:7],
        slp_var=era5_slp_var,
        tahiti_box=plot_cfg.regions.tahiti,
        darwin_box=plot_cfg.regions.darwin,
    ).to_pandas().dropna()

    # --- load model(s): can be multiple datasets
    for model_name in plot_cfg.models:
        model_cfg = cfg.datasets.models[model_name]

        # per-member SOI
        soi_members = {}
        for m in cfg.members:
            da_model = open_model_da(
                model_cfg=model_cfg,
                cfg = cfg,
                member=m,
                var=plot_cfg.variable,
                modelname = model_cfg.modelname,
                freq=plot_cfg.freq,
                start=start,
                end=end,
                grid=plot_cfg.get("grid", "gn"),
            )
            da_model = conversion_rules(plot_cfg.variable, da_model, cfg)
            ds_model = da_model.to_dataset(name=plot_cfg.variable)
            soi_members[m] = calc_soi(
                ds_model, start=start[:7], end=end[:7],
                slp_var=plot_cfg.variable,
                tahiti_box=plot_cfg.regions.tahiti,
                darwin_box=plot_cfg.regions.darwin,
            )

        if cfg.include_ensemble_mean_as_member:
            soi_members = ensemble_mean_as_member(soi_members, name="mean")

        # plot each member as separate fig (or only mean if you want)
        members_to_plot = plot_cfg.get("members_to_plot", None) or list(soi_members.keys())

        for mem in members_to_plot:
            model_series = soi_members[mem].to_pandas().dropna()

            # --- layout
            fig = plt.figure(figsize=tuple(plot_cfg.figsize))
            gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
            ax_ts = fig.add_subplot(gs[0, 0])
            ax_kde = fig.add_subplot(gs[0, 1], sharey=ax_ts) if plot_cfg.kde.enabled else None

            # model bars
            colors_model = np.where(model_series >= 0, "steelblue", "indianred")
            ax_ts.bar(model_series.index, model_series.values,
                      width=plot_cfg.bar_width_days,
                      color=colors_model, alpha=0.6, zorder=1)

            # ERA5 black line
            ax_ts.plot(soi_era5.index, soi_era5.values, color="black", linewidth=1.2, zorder=2)

            ax_ts.axhline(0, color="black", linewidth=1)
            ax_ts.set_xlabel("Time")
            ax_ts.set_ylabel("SOI")
            ax_ts.set_title(f"SOI: ERA5 vs {model_name} ({mem})", fontsize=14, weight="bold")
            ax_ts.grid(True, linestyle="--", alpha=0.3)

            # custom legend (split bar + black line)
            era5_handle = plt.Line2D([], [], color="black", linewidth=1.8, label="ERA5")
            model_handle = mpatches.Rectangle((0, 0), 1, 1, label=f"{model_name}")
            ax_ts.legend(
                handles=[era5_handle, model_handle],
                handler_map={model_handle: SplitBarHandler()},
                frameon=True,
                loc="upper left",
            )

            # KDE panel
            if plot_cfg.kde.enabled:
                y = np.linspace(
                    min(soi_era5.min(), model_series.min()) - plot_cfg.kde.pad,
                    max(soi_era5.max(), model_series.max()) + plot_cfg.kde.pad,
                    plot_cfg.kde.n_eval,
                )
                kde_era5 = gaussian_kde(soi_era5.values, bw_method=plot_cfg.kde.bw_method)
                kde_model = gaussian_kde(model_series.values, bw_method=plot_cfg.kde.bw_method)

                ax_kde.plot(kde_era5(y), y, color="black", linewidth=2, label="ERA5")
                ax_kde.plot(kde_model(y), y, color="steelblue", linewidth=2, label=model_name)
                ax_kde.axhline(0, color="black", linewidth=1)
                ax_kde.set_xlabel("Density")
                ax_kde.grid(True, linestyle="--", alpha=0.3)
                ax_kde.set_title("Kernel Density Estimation", fontsize=10, weight="bold")
                plt.setp(ax_kde.get_yticklabels(), visible=False)
                ax_kde.tick_params(axis="y", length=0)
                ax_kde.legend()

            # plt.show()
            if cfg.out.savefig:
                outdir = os.path.join(
                    hydra.utils.get_original_cwd(),
                    cfg.out.dir
                )
                os.makedirs(outdir, exist_ok=True)

                fname = "soi_kde.png" if cfg.plots.soi.kde.enabled else "soi.png"
                fig.savefig(
                    os.path.join(outdir, fname),
                    dpi=cfg.out.dpi,
                    bbox_inches="tight"
                )
                plt.close(fig)
            else:
                plt.show()