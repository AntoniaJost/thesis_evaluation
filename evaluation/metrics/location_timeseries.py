# evaluation/metrics/location_timeseries.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from evaluation.io import (
    ensure_allowed_var,
    resolve_period,
    open_model_da,
    open_era5_da,
    ensemble_mean_as_member,
    maybe_convert_units,
)


def run(cfg):
    plot_cfg = cfg.plots #.metrics.location_timeseries
    ensure_allowed_var(cfg, plot_cfg.variable)
    start, end = resolve_period(cfg, plot_cfg)

    var = plot_cfg.variable

    # ERA5
    era5_da = maybe_convert_units(var, open_era5_da(cfg, var, start, end), cfg)
    era5_point = era5_da.sel(lat=plot_cfg.location.lat, lon=plot_cfg.location.lon, method="nearest")

    for model_name in plot_cfg.models:
        model_cfg = cfg.datasets.models[model_name]

        members = {}
        for m in cfg.members:
            da = maybe_convert_units(var, open_model_da(model_cfg, m, var, plot_cfg.freq, start, end, grid=plot_cfg.grid), cfg)
            members[m] = da.sel(lat=plot_cfg.location.lat, lon=plot_cfg.location.lon, method="nearest")

        if cfg.include_ensemble_mean_as_member and plot_cfg.include_mean_member:
            members = ensemble_mean_as_member(members, name="mean")

        fig, ax = plt.subplots(figsize=tuple(plot_cfg.figsize))

        ax.plot(era5_point.time.values, era5_point.values, color="black", lw=1.4, label="ERA5")

        # plot members
        for i, (m, da) in enumerate(members.items()):
            lw = 1.6 if m == "mean" else 0.8
            a = 0.9 if m == "mean" else 0.25
            ax.plot(da.time.values, da.values, lw=lw, alpha=a, label=f"{model_name} {m}")

        ax.set_title(plot_cfg.title.format(var=var, model=model_name), fontsize=13, weight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel(plot_cfg.ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize="small", frameon=False)
        plt.show()
