# evaluation/metrics/anomalies.py
from __future__ import annotations

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import hydra

from evaluation.general_functions import (
    model_abbrev,
    open_model_da,
    open_era5_da,
    conversion_rules,
    should_compute_output,
    iter_vars_and_plevs,
    plev_strings,
    detrend_dataarray,
    format_unit_for_plot
)

from evaluation.metrics.global_mean import (
    lin_reg,
    trend_decay,
    annual_weighted_mean,
    area_weighted_global_mean,
    select_colour,
    select_light_colour,
)


def to_anomaly(da: xr.DataArray, baseline_start: str, baseline_end: str):
    base = da.sel(time=slice(baseline_start, baseline_end)).mean("time")
    if base.ndim == 0:
        base_info = float(base.values)
    else:
        base_info = base
    return da - base, base_info


def _mode(plot_cfg) -> str:
    mode = str(getattr(plot_cfg, "mode", "anomaly")).strip().lower()
    allowed = {"anomaly", "detrend", "both"}
    if mode not in allowed:
        raise ValueError(
            f"plots.anomalies.mode must be one of {sorted(allowed)}. Got: {mode}"
        )
    return mode


def _detrend_bounds(plot_cfg, start: str, end: str) -> tuple[str, str]:
    """
    determine which period is used to fit the detrending line
    if detr_baseline.start/end are null, use the full selected period
    """
    detr_start = plot_cfg.detr_baseline.start
    detr_end = plot_cfg.detr_baseline.end

    if detr_start in (None, "null") or detr_end in (None, "null"):
        return start, end

    return str(detr_start), str(detr_end)


def _apply_mode_to_series(da: xr.DataArray, plot_cfg, start: str, end: str) -> xr.DataArray:
    """
    apply anomaly / detrend / both to annual-mean series
    order for mode='both': detrend first, then anomaly
    """
    mode = _mode(plot_cfg)

    if mode in {"detrend", "both"}:
        detr_start, detr_end = _detrend_bounds(plot_cfg, start, end)
        da = detrend_dataarray(
            da,
            dim="time",
            start=detr_start,
            end=detr_end,
            preserve_mean=bool(plot_cfg.detrend.preserve_mean),
        )

    if mode in {"anomaly", "both"}:
        da, _ = to_anomaly(
            da,
            plot_cfg.anom_baseline.start,
            plot_cfg.anom_baseline.end,
        )

    return da


def _mode_tag(plot_cfg) -> str:
    mode = _mode(plot_cfg)
    if mode == "anomaly":
        return "anom"
    if mode == "detrend":
        return "detrended-presvMeanTrue" if plot_cfg.detrend.preserve_mean else "detrended-presvMeanFalse"
    if mode == "both":
        return "anom_detrended-presvMeanTrue" if plot_cfg.detrend.preserve_mean else "anom_detrended-presvMeanFalse"
    return mode


def _mode_title_suffix(plot_cfg, start: str, end: str) -> str:
    # extra title lines describing anomaly / detrending settings
    mode = _mode(plot_cfg)
    lines = []

    if mode in {"anomaly", "both"}:
        a0 = str(plot_cfg.anom_baseline.start)[:7]
        a1 = str(plot_cfg.anom_baseline.end)[:7]
        lines.append(f"Baseline removed: mean of {a0} – {a1}")

    if mode in {"detrend", "both"}:
        d0, d1 = _detrend_bounds(plot_cfg, start, end)
        if plot_cfg.detrend.preserve_mean:
            text = "Linear slope removed"
        else:
            text = "Full trend removed"

        if d0 == start and d1 == end:
            lines.append(f"{text} over entire time period")
        else:
            lines.append(f"{text} over {str(d0)[:7]} – {str(d1)[:7]}")

    return "\n".join(lines)


def run(cfg):
    plot_cfg = cfg.plots.anomalies #.metrics.anomalies
    add_dir = str(plot_cfg.special_outdir) if plot_cfg.special_outdir else ""
    mode = _mode(plot_cfg)

    for item in iter_vars_and_plevs(cfg, plot_cfg):
        var = item["var"]
        long_name = item["long_name"]
        unit = format_unit_for_plot(item["unit"])
        start = item["start"]
        end = item["end"]

        for plev in item["plevs"]:
            # check if file already exists and if recomputation is wanted
            plev_title, plev_tag = plev_strings(plev)
            outdir = os.path.join(
                hydra.utils.get_original_cwd(),
                cfg.out.dir,
                "anomalies",
                add_dir
            )
            os.makedirs(outdir, exist_ok=True)

            model_tags = "_".join(model_abbrev(m) for m in plot_cfg.models)
            start_tag = start.replace("-", "")
            end_tag = end.replace("-", "")
            mode_tag = _mode_tag(plot_cfg)
            fname = f"anomalies_{var}{plev_tag}_{model_tags}_{start_tag}-{end_tag}_{mode_tag}.png"
            outfile = os.path.join(outdir, fname)
            if cfg.out.savefig and not should_compute_output(outfile, getattr(cfg.out, "overwrite", "ask")):
                continue

            unit_here = unit
            # compute ERA5 annual GM
            era5_da = open_era5_da(cfg, var=var, start=start, end=end, plev=plev, freq=plot_cfg.freq, grid=plot_cfg.grid)
            era5_source = "era5_cmor" if plot_cfg.freq == "daily" else "era5_natural"
            era5_da, unit_here = conversion_rules(var, era5_da, cfg, era5_source, unit_here)
            gm_era5 = area_weighted_global_mean(era5_da)
            agm_era5 = annual_weighted_mean(gm_era5)

            # anomalies relative to baseline
            series_era5 = _apply_mode_to_series(agm_era5, plot_cfg, start, end)
            lrg_era5_anom, slope_era5 = lin_reg(series_era5)
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
                    series = _apply_mode_to_series(agm, plot_cfg, start, end)
                    member_anom[m] = series
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

            years = series_era5.time.dt.year.values
            ax.plot(years, series_era5.values, color="black", linewidth=1.5,
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

            draw_zero = (
                mode in {"anomaly", "both"}
                or (mode == "detrend" and not plot_cfg.detrend.preserve_mean)
            )
            if draw_zero:
                ax.axhline(0, color="0.3", linewidth=1.0, alpha=0.35)

            if plot_cfg.title:
                title = plot_cfg.title.format(
                    var=var,
                    long_name=long_name,
                    anom_base_start=str(plot_cfg.anom_baseline.start)[:7],
                    anom_base_end=str(plot_cfg.anom_baseline.end)[:7],
                    detr_base_start=str(_detrend_bounds(plot_cfg, start, end)[0])[:7],
                    detr_base_end=str(_detrend_bounds(plot_cfg, start, end)[1])[:7],
                    mode=mode,
                )
            else:
                if mode == "anomaly":
                    title = f"Global Annual Mean {long_name} Anomaly ({var}{plev_title})"
                elif mode == "detrend":
                    title = f"Global Annual Mean {long_name} ({var}{plev_title})"
                else:  # both
                    title = f"Global Annual Mean {long_name} Anomaly ({var}{plev_title})"

            suffix = _mode_title_suffix(plot_cfg, start, end)
            if suffix:
                title += f"\n{suffix}"

            ax.set_title(f"{title}", fontsize=12, pad=8) #\n({start} – {end})
            ax.set_xlabel("Year")
            if mode == "anomaly":
                ylabel = f"Anomaly in {long_name} ({unit_here})"
            elif mode == "detrend":
                ylabel = f"{long_name} ({unit_here})"
            else:  # both
                ylabel = f"Anomaly in {long_name} ({unit_here})"
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(plot_cfg.ticks.major))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(plot_cfg.ticks.minor))
            ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
            ax.grid(False, which="minor")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.margins(x=0.01)

            # ax.legend(loc=plot_cfg.legend.loc, fontsize="x-small", frameon=False, handlelength=2.6, borderaxespad=0.6)
            if plot_cfg.legend.inside_plot:
                ax.legend(loc=plot_cfg.legend.loc, frameon=False, fontsize="x-small", borderaxespad=0.0)
            else:
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize="x-small")
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