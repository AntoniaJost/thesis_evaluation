# evaluation/metrics/anomalies.py
from __future__ import annotations

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import hydra

from evaluation.general_functions import (
    ensure_allowed_var,
    resolve_period,
    normalise_vars,
    plevs_for_variable,
    open_model_da_raw,
    open_model_da,
    open_era5_da,
    conversion_rules,
    should_compute_output,
)

from evaluation.metrics.global_mean import (
    lin_reg,
    trend_decay,
    annual_weighted_mean,
    area_weighted_global_mean,
    select_colour,
    select_light_colour,
    model_abbrev,
)


def to_anomaly(da: xr.DataArray, baseline_start: str, baseline_end: str):
    base = da.sel(time=slice(baseline_start, baseline_end)).mean("time")
    return da - base, float(base.values)


def run(cfg):
    plot_cfg = cfg.plots.anomalies #.metrics.anomalies
    start, end = resolve_period(cfg, plot_cfg)

    vars_to_plot = normalise_vars(plot_cfg.variable)
    requested_plevs = getattr(plot_cfg, "plev", None)
    
    base_start = plot_cfg.baseline.start
    base_end = plot_cfg.baseline.end    

    for var in vars_to_plot:
        ensure_allowed_var(cfg, var)
        meta = cfg.variables.meta.get(var, None)
        long_name = meta.long_name if meta else var
        unit = meta.unit if meta else ""

        # inspect one sample model file to determine whether this variable has plev
        sample_model_name = plot_cfg.models[0]
        sample_model_cfg = cfg.datasets.models[sample_model_name]
        sample_member = cfg.members[0]

        da_sample = open_model_da_raw(
            sample_model_cfg,
            cfg,
            sample_member,
            var,
            sample_model_cfg.modelname,
            plot_cfg.freq,
            start,
            end,
            grid=plot_cfg.grid,
        )

        plevs = plevs_for_variable(da_sample, requested_plevs)

        for plev in plevs:
            # check if file already exists and if recomputation is wanted
            plev_title = ""
            plev_tag = ""
            if plev is not None:
                plev_pa = int(float(plev) * 100) if float(plev) < 2000 else int(float(plev))
                plev_hpa = int(plev_pa / 100)
                plev_title = f" at {plev_hpa} hPa"
                plev_tag = f"@{plev_hpa}hPa"

            outdir = os.path.join(
                hydra.utils.get_original_cwd(),
                cfg.out.dir,
                "anomalies"
            )
            os.makedirs(outdir, exist_ok=True)

            model_tags = "_".join(model_abbrev(m) for m in plot_cfg.models)
            start_tag = start.replace("-", "")
            end_tag = end.replace("-", "")
            fname = f"anomalies_{var}{plev_tag}_{model_tags}_{start_tag}-{end_tag}.png"
            outfile = os.path.join(outdir, fname)
            if cfg.out.savefig and not should_compute_output(outfile, getattr(cfg.out, "overwrite", "ask")):
                continue

            unit_here = unit
            # compute ERA5 annual GM
            era5_da = open_era5_da(cfg, var=var, start=start, end=end, plev=plev)
            era5_da, unit_here = conversion_rules(var, era5_da, cfg, "era5", unit_here)
            gm_era5 = area_weighted_global_mean(era5_da)
            agm_era5 = annual_weighted_mean(gm_era5)

            # anomalies relative to baseline
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
                    da = open_model_da(model_cfg, cfg, m, var, model_cfg.modelname, plot_cfg.freq, start, end, grid=plot_cfg.grid, plev=plev)
                    da, _ = conversion_rules(var, da, cfg, "model", unit_here)
                    gm = area_weighted_global_mean(da)
                    agm = annual_weighted_mean(gm)
                    anom, _ = to_anomaly(agm, base_start, base_end)
                    member_anom[m] = anom
                anom_members[model_name] = member_anom

                member_names = sorted(member_anom.keys())
                da_members = xr.concat([member_anom[m] for m in member_names], dim="member").assign_coords(member=member_names)

                stats_ds = xr.Dataset({
                    "min": da_members.min("member"),
                    "max": da_members.max("member"),
                    "mean": da_members.mean("member")})
                minmax_ds[model_name] = stats_ds

                lrg, slope = lin_reg(stats_ds["mean"])
                lrg_mean_ens[model_name] = lrg
                trend_mean_ens[model_name] = trend_decay(slope)

            # plot
            fig, ax = plt.subplots(figsize=tuple(plot_cfg.figsize), constrained_layout=True)

            years = anom_era5.time.dt.year.values
            ax.plot(years, anom_era5.values, color="black", linewidth=1.5,
                    label=f"ERA5 (Trend = {trend_era5} {unit_here}/dec)", zorder=6)
            ax.plot(years, lrg_era5_anom, color="black", linestyle="--", linewidth=1.0, zorder=7)

            for model_name in plot_cfg.models:
                c = select_colour(model_name, plot_cfg)
                for _, anom in anom_members[model_name].items():
                    yrs = anom.time.dt.year.values
                    ax.plot(yrs, anom.values, color=c, alpha=0.20, linewidth=0.8, zorder=2)

            for model_name in plot_cfg.models:
                c = select_colour(model_name, plot_cfg)
                fill = select_light_colour(model_name, plot_cfg)
                ds = minmax_ds[model_name]
                yrs = ds.time.dt.year.values

                ax.fill_between(yrs, ds["min"].values, ds["max"].values,
                                color=fill, alpha=0.25, zorder=1,
                                label=f"{cfg.datasets.models[model_name].proper_name} (Ens. trend: {trend_mean_ens[model_name]} {unit_here}/dec)")
                ax.plot(yrs, ds["mean"].values, color=c, linewidth=1.2, zorder=4)
                ax.plot(yrs, lrg_mean_ens[model_name], color=c, linestyle="--", linewidth=1.0, zorder=5)

            ax.axhline(0, color="0.3", linewidth=1.0, alpha=0.35)

            if plot_cfg.title:
                title = plot_cfg.title.format(
                    var=var,
                    long_name=long_name,
                    base_start=base_start[:7],
                    base_end=base_end[:7],
                )
            else:
                title = f"Global Annual Mean {long_name} Anomaly ({var}{plev_title})\nBaseline removed: mean of ({base_start[:7]}–{base_end[:7]})"

            ax.set_title(f"{title}", fontsize=12, pad=8) #\n({start} – {end})
            ax.set_xlabel("Year")
            ax.set_ylabel(f"Anomaly in {long_name} ({unit_here})")

            ax.xaxis.set_major_locator(mticker.MultipleLocator(plot_cfg.ticks.major))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(plot_cfg.ticks.minor))
            ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
            ax.grid(True, which="minor", linewidth=0.5, alpha=0.2)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.margins(x=0.01)

            ax.legend(loc=plot_cfg.legend.loc, fontsize="x-small", frameon=False, handlelength=2.6, borderaxespad=0.6)
            # plt.show()
            if cfg.out.savefig:
                fig.savefig(
                    os.path.join(outdir, fname),
                    dpi=cfg.out.dpi,
                    bbox_inches="tight"
                )
                plt.close(fig)
            else:
                plt.show()