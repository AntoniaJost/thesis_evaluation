from __future__ import annotations

import os
import calendar
from collections import OrderedDict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D

from evaluation.general_functions import (
    ensure_allowed_var,
    iter_vars_and_plevs,
    open_model_da,
    open_era5_da,
    conversion_rules,
    should_compute_output,
    model_abbrev,
    ensemble_mean_as_member,
    plev_strings,
    format_unit_for_plot
)
from evaluation.metrics.individual_plots import (
    _area_mean,
    _subtract_with_time_alignment,
    _wrap_lon_360
)
from evaluation.metrics.seasonal_cycle import (
    SEASON_MONTHS,
    _selected_seasons,
    _subset_for_region,
    _region_tag_and_label
)
from evaluation.metrics.zonal_mean import(
    _format_lat,
    _format_lon
)


def _resolve_figsize(plot_cfg):
    if plot_cfg.figsize not in (None, "null"):
        return tuple(plot_cfg.figsize)
    return (11, 6)


def _month_order(season: str) -> list[int]:
    if season == "DJF":
        return [12, 1, 2]
    return list(SEASON_MONTHS[season])


def _month_labels(months: list[int]) -> list[str]:
    return [calendar.month_abbr[m] for m in months]


def _pretty_region_label(plot_cfg, region_label: str) -> str:
    """
    for individual area selection, add nice coordinate formatting for title
    """
    if str(getattr(plot_cfg, "region", "global")).strip().lower() != "individual":
        return region_label

    lat0 = float(plot_cfg.individual.lat0)
    lat1 = float(plot_cfg.individual.lat1)
    lon0 = _wrap_lon_360(float(plot_cfg.individual.lon0))
    lon1 = _wrap_lon_360(float(plot_cfg.individual.lon1))

    lat0_label = _format_lat(lat0)
    lat1_label = _format_lat(lat1)
    lon0_label = _format_lon(lon0)
    lon1_label = _format_lon(lon1)

    return f"Area: {lat0_label} to {lat1_label}, {lon0_label} to {lon1_label}"


def _prepare_time_series(series: xr.DataArray, freq: str) -> xr.DataArray:
    if "time" not in series.dims:
        raise ValueError("Expected a time-dependent series.")

    freq = str(freq).strip().lower()

    if freq == "daily":
        out = series.sortby("time")
    elif freq == "monthly":
        month_start = pd.to_datetime(series["time"].dt.strftime("%Y-%m-01").values)
        out = series.copy().assign_coords(time=month_start).sortby("time")
    else:
        raise ValueError(f"Unsupported freq='{freq}'. Expected 'daily' or 'monthly'.")

    return out


def _prepare_series(da: xr.DataArray, plot_cfg) -> xr.DataArray:
    da_region = _subset_for_region(da, plot_cfg)
    series = _area_mean(da_region)
    return _prepare_time_series(series, plot_cfg.freq)


def _to_cycle_anomaly(series: xr.DataArray, baseline_start: str, baseline_end: str, freq: str) -> xr.DataArray:
    """
    calc anomaly
    - monthly data -> subtract baseline monthly climatology
    - daily data   -> subtract baseline day-of-year climatology
    """
    base = series.sel(time=slice(baseline_start, baseline_end))
    if base.sizes.get("time", 0) == 0:
        raise ValueError(
            f"No baseline data found between {baseline_start} and {baseline_end}."
        )

    freq = str(freq).strip().lower()

    if freq == "monthly":
        clim = base.groupby("time.month").mean("time")
        return series.groupby("time.month") - clim

    if freq == "daily":
        clim = base.groupby("time.dayofyear").mean("time")
        return series.groupby("time.dayofyear") - clim

    raise ValueError(f"Unsupported freq='{freq}'. Expected 'daily' or 'monthly'.")


def _apply_processing(
    series_model: xr.DataArray,
    plot_cfg,
    baseline_start: str,
    baseline_end: str,
    series_era5: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    4 modes via 2 booleans:
      anomaly=False, difference=False -> absolute model monthly cycle
      anomaly=True,  difference=False -> model monthly anomaly cycle
      anomaly=False, difference=True  -> model - ERA5 monthly cycle
      anomaly=True,  difference=True  -> (model anom) - (ERA5 anom)
    """
    anomaly = bool(plot_cfg.anomaly)
    difference = bool(plot_cfg.difference)

    model_proc = series_model
    era5_proc = series_era5

    if anomaly:
        model_proc = _to_cycle_anomaly(
            model_proc,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            freq=plot_cfg.freq,
        )
        if difference:
            if era5_proc is None:
                raise ValueError("difference=True requires ERA5 data.")
            era5_proc = _to_cycle_anomaly(
                era5_proc,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                freq=plot_cfg.freq,
            )

    if difference:
        if era5_proc is None:
            raise ValueError("difference=True requires ERA5 data.")
        return _subtract_with_time_alignment(model_proc, era5_proc, plot_cfg)

    return model_proc


def _build_time_lines(series: xr.DataArray, season: str, freq: str):
    s = series.to_pandas().dropna()
    freq = str(freq).strip().lower()

    if freq == "monthly":
        months_order = _month_order(season)
        s = s[s.index.month.isin(months_order)]
        grouped = s.groupby([s.index.year, s.index.month]).mean()

        out = OrderedDict()
        years = sorted({y for y, _ in grouped.index})

        for year in years:
            xs = []
            vals = []
            for i, m in enumerate(months_order, start=1):
                key = (year, m)
                if key in grouped.index:
                    xs.append(i)
                    vals.append(float(grouped.loc[key]))
            if vals:
                out[year] = (xs, np.asarray(vals, dtype=float))

        xticks = list(range(1, len(months_order) + 1))
        xticklabels = _month_labels(months_order)
        return out, xticks, xticklabels

    if freq == "daily":
        if season == "full":
            s = s.copy()
        else:
            months_here = set(SEASON_MONTHS[season])
            s = s[s.index.month.isin(months_here)]

        grouped = s.groupby([s.index.year, s.index.dayofyear]).mean()

        out = OrderedDict()
        years = sorted({y for y, _ in grouped.index})

        for year in years:
            sub = grouped.loc[year]
            xs = sub.index.to_list()
            vals = sub.to_numpy(dtype=float)
            if len(vals) > 0:
                out[year] = (xs, vals)

        if season == "full":
            xticks = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
            xticklabels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        else:
            xticks = None
            xticklabels = None

        return out, xticks, xticklabels

    raise ValueError(f"Unsupported freq='{freq}'. Expected 'daily' or 'monthly'.")


def _year_group_label(year: int, min_year: int, max_year: int, group_size: int) -> str:
    start = min_year + ((year - min_year) // group_size) * group_size
    end = min(start + group_size - 1, max_year)
    return f"{start}–{end}"


def _group_palette(n: int) -> list[str]:
    """
    Colour-blind-friendly palette.
    reuses Okabe-Ito-friendly colours and extends with Tol-style safe colours
    """
    base = [
        "#332288",  # dark blue
        "#88CCEE",  # light blue
        "#44AA99",  # teal
        "#117733",  # green
        "#DDCC77",  # sand
        "#CC6677",  # rose
        "#AA4499",  # purple
        "#882255",  # wine
    ]
    if n <= len(base):
        return base[:n]
    reps = (n // len(base)) + 1
    return (base * reps)[:n]


def _group_colour_map(years: list[int], group_size: int) -> tuple[dict[int, str], list[tuple[str, str]]]:
    min_year = min(years)
    max_year = max(years)

    labels = []
    year_to_label = {}
    for y in years:
        label = _year_group_label(y, min_year, max_year, group_size)
        year_to_label[y] = label
        if label not in labels:
            labels.append(label)

    colours = _group_palette(len(labels))
    label_to_colour = {lab: col for lab, col in zip(labels, colours)}
    year_to_colour = {y: label_to_colour[year_to_label[y]] for y in years}

    legend_items = [(lab, label_to_colour[lab]) for lab in labels]
    return year_to_colour, legend_items


def _mode_tag(plot_cfg) -> str:
    if plot_cfg.anomaly and plot_cfg.difference:
        return "anom_minusERA5"
    if plot_cfg.anomaly:
        return "anom"
    if plot_cfg.difference:
        return "minusERA5"
    return "abs"


def _ylabel(long_name: str, unit_here: str, plot_cfg) -> str:
    if plot_cfg.anomaly and plot_cfg.difference:
        return f"{long_name} anomaly difference to ERA5 ({unit_here})"
    if plot_cfg.anomaly:
        return f"{long_name} anomaly ({unit_here})"
    if plot_cfg.difference:
        return f"{long_name} difference to ERA5 ({unit_here})"
    return f"{long_name} ({unit_here})"


def _default_title(
    long_name: str,
    plev_title: str,
    proper_model_name: str,
    member: str,
    region_label: str,
    season: str,
    start: str,
    end: str,
    plot_cfg,
) -> str:
    season_label = "Full Year" if season == "full" else season
    freq_label = "Daily" if str(plot_cfg.freq).lower() == "daily" else "Monthly"

    if plot_cfg.anomaly and plot_cfg.difference:
        lead = f"{freq_label} {long_name}{plev_title} Anomaly Difference to ERA5"
    elif plot_cfg.anomaly:
        lead = f"{freq_label} {long_name}{plev_title} Anomaly"
    elif plot_cfg.difference:
        lead = f"{freq_label} {long_name}{plev_title} Difference to ERA5"
    else:
        lead = f"{freq_label} {long_name}{plev_title}"

    title = f"{lead}\n{proper_model_name} {member} | {season_label} | {region_label} | {start} – {end}"

    if plot_cfg.anomaly:
        title += (
            f"\nBaseline: {str(plot_cfg.baseline.start)[:10]} – {str(plot_cfg.baseline.end)[:10]}"
        )

    return title


def _date_tag(date_str: str, freq: str) -> str:
    ts = pd.to_datetime(date_str)
    if str(freq).lower() == "monthly":
        return ts.strftime("%Y%m")
    return ts.strftime("%Y%m%d")


def _output_filename(
    var: str,
    plev_tag: str,
    model_tag: str,
    member: str,
    region_tag: str,
    season: str,
    start: str,
    end: str,
    mode_tag: str,
    freq: str,
) -> str:
    start_tag = _date_tag(start, freq)
    end_tag = _date_tag(end, freq)
    return (
        f"yearly_cycle_{var}{plev_tag}_{model_tag}{member}_{region_tag}_{season}_"
        f"{start_tag}-{end_tag}_{mode_tag}.png"
    )


def run(cfg):
    plot_cfg = cfg.plots.yearly_cycle
    add_dir = str(plot_cfg.special_outdir) if plot_cfg.special_outdir else ""
    seasons = _selected_seasons(plot_cfg)
    figsize = _resolve_figsize(plot_cfg)

    if bool(plot_cfg.map_era5) and bool(plot_cfg.difference):
        raise ValueError(
            "plots.yearly_cycle.map_era5=true cannot be combined with difference=true, "
            "because ERA5-to-ERA5 differences are not meaningful."
        )

    for item in iter_vars_and_plevs(cfg, plot_cfg):
        var = item["var"]
        long_name = item["long_name"]
        unit = format_unit_for_plot(item["unit"])
        start = item["start"]
        end = item["end"]

        ensure_allowed_var(cfg, var)

        for plev in item["plevs"]:
            plev_title, plev_tag = plev_strings(plev)
            region_tag, region_label = _region_tag_and_label(plot_cfg)
            region_label_pretty = _pretty_region_label(plot_cfg, region_label)

            # --- MODEL PLOTS ---
            for model_name in plot_cfg.models:
                model_cfg = cfg.datasets.models[model_name]
                proper_model_name = getattr(model_cfg, "proper_name", model_name)
                model_tag = model_abbrev(model_name)

                members_to_plot = list(cfg.members)
                if plot_cfg.include_ensemble_mean_as_member:
                    members_to_plot = list(cfg.members) + ["mean"]

                if plot_cfg.only_mean:
                    if not plot_cfg.include_ensemble_mean_as_member:
                        raise ValueError(
                            "plots.yearly_cycle.only_mean=true requires "
                            "include_ensemble_mean_as_member=true."
                        )
                    members_to_plot = ["mean"]

                for member in members_to_plot:
                    for season in seasons:
                        outdir = os.path.join(
                            hydra.utils.get_original_cwd(),
                            cfg.out.dir,
                            "yearly_cycle",
                            var,
                            add_dir,
                        )
                        os.makedirs(outdir, exist_ok=True)

                        outfile = os.path.join(
                            outdir,
                            _output_filename(
                                var=var,
                                plev_tag=plev_tag,
                                model_tag=model_tag,
                                member=f"_{str(member)}",
                                region_tag=region_tag,
                                season=season,
                                start=start,
                                end=end,
                                mode_tag=_mode_tag(plot_cfg),
                                freq=plot_cfg.freq,
                            ),
                        )

                        if cfg.out.savefig and not should_compute_output(
                            outfile, getattr(cfg.out, "overwrite", "ask")
                        ):
                            continue

                        unit_here = unit

                        era5_series = None
                        if plot_cfg.difference:
                            era5_da = open_era5_da(
                                cfg,
                                var=var,
                                start=start,
                                end=end,
                                plev=plev,
                                freq=plot_cfg.freq,
                                grid=plot_cfg.grid,
                            )
                            era5_source = (
                                "era5_cmor"
                                if str(plot_cfg.freq).lower() == "daily"
                                else "era5_natural"
                            )
                            era5_da, unit_here = conversion_rules(
                                var, era5_da, cfg, era5_source, unit_here
                            )
                            era5_series = _prepare_series(era5_da, plot_cfg)

                        member_to_series = {}

                        for real_member in cfg.members:
                            da_model = open_model_da(
                                model_cfg=model_cfg,
                                cfg=cfg,
                                member=real_member,
                                var=var,
                                modelname=model_cfg.modelname,
                                freq=plot_cfg.freq,
                                start=start,
                                end=end,
                                grid=plot_cfg.grid,
                                plev=plev,
                            )
                            da_model, unit_here = conversion_rules(
                                var, da_model, cfg, "model", unit_here
                            )
                            member_to_series[real_member] = _prepare_series(
                                da_model, plot_cfg
                            )

                        if plot_cfg.include_ensemble_mean_as_member:
                            member_to_series = ensemble_mean_as_member(
                                member_to_series, name="mean"
                            )

                        if member not in member_to_series:
                            raise ValueError(
                                f"Requested member '{member}' not available in member_to_series."
                            )

                        model_series = member_to_series[member]

                        processed = _apply_processing(
                            series_model=model_series,
                            plot_cfg=plot_cfg,
                            baseline_start=plot_cfg.baseline.start,
                            baseline_end=plot_cfg.baseline.end,
                            series_era5=era5_series,
                        )

                        year_lines, xticks, xticklabels = _build_time_lines(
                            processed, season=season, freq=plot_cfg.freq
                        )
                        if len(year_lines) == 0:
                            raise ValueError(
                                f"No data left to plot for {model_name} {member}, season={season}."
                            )

                        years = list(year_lines.keys())
                        year_to_colour, legend_items = _group_colour_map(
                            years, int(plot_cfg.group_years)
                        )

                        fig, ax = plt.subplots(
                            figsize=figsize, constrained_layout=True
                        )

                        for year, (xs, vals) in year_lines.items():
                            ax.plot(
                                xs,
                                vals,
                                color=year_to_colour[year],
                                linewidth=float(plot_cfg.linewidth),
                                alpha=float(plot_cfg.alpha),
                                zorder=2,
                            )

                        if bool(plot_cfg.highlight_previous_year) and len(years) > 1:
                            prev = sorted(years)[-2]
                            xs, vals = year_lines[prev]
                            ax.plot(
                                xs,
                                vals,
                                color="#D55E00",
                                linewidth=float(plot_cfg.previous_linewidth),
                                alpha=1.0,
                                zorder=3,
                                label=str(prev),
                            )

                        if bool(plot_cfg.highlight_latest_year) and len(years) > 0:
                            latest = max(years)
                            xs, vals = year_lines[latest]
                            ax.plot(
                                xs,
                                vals,
                                color="black",
                                linewidth=float(plot_cfg.latest_linewidth),
                                alpha=1.0,
                                zorder=5,
                                label=str(latest),
                            )

                        if plot_cfg.anomaly or plot_cfg.difference:
                            ax.axhline(
                                0.0,
                                color="0.35",
                                linewidth=1.0,
                                alpha=0.7,
                                zorder=1,
                            )

                        if plot_cfg.title:
                            title = str(plot_cfg.title).format(
                                var=var,
                                long_name=long_name,
                                model=proper_model_name,
                                member=member,
                                region=region_label_pretty,
                                season=season,
                                start=start,
                                end=end,
                                baseline_start=str(plot_cfg.baseline.start)[:10],
                                baseline_end=str(plot_cfg.baseline.end)[:10],
                            )
                        else:
                            title = _default_title(
                                long_name=long_name,
                                plev_title=plev_title,
                                proper_model_name=proper_model_name,
                                member=str(member),
                                region_label=region_label_pretty,
                                season=season,
                                start=start,
                                end=end,
                                plot_cfg=plot_cfg,
                            )

                        ax.set_title(title, pad=10)
                        ax.set_ylabel(_ylabel(long_name, unit_here, plot_cfg))
                        ax.set_xlabel(
                            "Month"
                            if str(plot_cfg.freq).lower() == "monthly"
                            else "Day of year"
                        )

                        if xticks is not None and xticklabels is not None:
                            ax.set_xticks(xticks)
                            ax.set_xticklabels(xticklabels)

                        ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        ax.margins(x=0.01)

                        handles = [
                            Line2D([0], [0], color=col, lw=2.2, label=lab)
                            for lab, col in legend_items
                        ]

                        if bool(plot_cfg.highlight_previous_year) and len(years) > 1:
                            handles.append(
                                Line2D(
                                    [0],
                                    [0],
                                    color="#D55E00",
                                    lw=float(plot_cfg.previous_linewidth),
                                    label=str(sorted(years)[-2]),
                                )
                            )

                        if bool(plot_cfg.highlight_latest_year) and len(years) > 0:
                            handles.append(
                                Line2D(
                                    [0],
                                    [0],
                                    color="black",
                                    lw=float(plot_cfg.latest_linewidth),
                                    label=str(max(years)),
                                )
                            )

                        if plot_cfg.legend.inside_plot:
                            ax.legend(
                                handles=handles,
                                loc=plot_cfg.legend.loc,
                                frameon=False,
                                fontsize="small",
                                borderaxespad=0.0,
                            )
                        else:
                            ax.legend(
                                handles=handles,
                                loc="upper center",
                                bbox_to_anchor=(0.5, -0.15),
                                ncol=min(4, max(1, len(handles))),
                                frameon=False,
                                fontsize="small",
                            )

                        if cfg.out.savefig:
                            fig.savefig(outfile, dpi=cfg.out.dpi, bbox_inches="tight")
                            plt.close(fig)
                        else:
                            plt.show()

            # --- OPTIONAL ERA5 PLOTS ----
            if bool(plot_cfg.map_era5):
                for season in seasons:
                    outdir = os.path.join(
                        hydra.utils.get_original_cwd(),
                        cfg.out.dir,
                        "yearly_cycle",
                        var,
                        add_dir,
                    )
                    os.makedirs(outdir, exist_ok=True)

                    outfile = os.path.join(
                        outdir,
                        _output_filename(
                            var=var,
                            plev_tag=plev_tag,
                            model_tag="era5",
                            member="",
                            region_tag=region_tag,
                            season=season,
                            start=start,
                            end=end,
                            mode_tag=_mode_tag(plot_cfg),
                            freq=plot_cfg.freq,
                        ),
                    )

                    if cfg.out.savefig and not should_compute_output(
                        outfile, getattr(cfg.out, "overwrite", "ask")
                    ):
                        continue

                    unit_here = unit

                    era5_da = open_era5_da(
                        cfg,
                        var=var,
                        start=start,
                        end=end,
                        plev=plev,
                        freq=plot_cfg.freq,
                        grid=plot_cfg.grid,
                    )
                    era5_source = (
                        "era5_cmor"
                        if str(plot_cfg.freq).lower() == "daily"
                        else "era5_natural"
                    )
                    era5_da, unit_here = conversion_rules(
                        var, era5_da, cfg, era5_source, unit_here
                    )
                    era5_series = _prepare_series(era5_da, plot_cfg)

                    # for ERA5 plots, only allow absolute or anomaly (era5-era5=0)
                    if plot_cfg.anomaly:
                        processed = _to_cycle_anomaly(
                            era5_series,
                            baseline_start=plot_cfg.baseline.start,
                            baseline_end=plot_cfg.baseline.end,
                            freq=plot_cfg.freq,
                        )
                    else:
                        processed = era5_series

                    year_lines, xticks, xticklabels = _build_time_lines(
                        processed, season=season, freq=plot_cfg.freq
                    )
                    if len(year_lines) == 0:
                        raise ValueError(
                            f"No ERA5 data left to plot for season={season}."
                        )

                    years = list(year_lines.keys())
                    year_to_colour, legend_items = _group_colour_map(
                        years, int(plot_cfg.group_years)
                    )

                    fig, ax = plt.subplots(
                        figsize=figsize, constrained_layout=True
                    )

                    for year, (xs, vals) in year_lines.items():
                        ax.plot(
                            xs,
                            vals,
                            color=year_to_colour[year],
                            linewidth=float(plot_cfg.linewidth),
                            alpha=float(plot_cfg.alpha),
                            zorder=2,
                        )

                    if bool(plot_cfg.highlight_previous_year) and len(years) > 1:
                        prev = sorted(years)[-2]
                        xs, vals = year_lines[prev]
                        ax.plot(
                            xs,
                            vals,
                            color="#D55E00",
                            linewidth=float(plot_cfg.previous_linewidth),
                            alpha=1.0,
                            zorder=3,
                            label=str(prev),
                        )

                    if bool(plot_cfg.highlight_latest_year) and len(years) > 0:
                        latest = max(years)
                        xs, vals = year_lines[latest]
                        ax.plot(
                            xs,
                            vals,
                            color="black",
                            linewidth=float(plot_cfg.latest_linewidth),
                            alpha=1.0,
                            zorder=5,
                            label=str(latest),
                        )

                    if plot_cfg.anomaly:
                        ax.axhline(
                            0.0,
                            color="0.35",
                            linewidth=1.0,
                            alpha=0.7,
                            zorder=1,
                        )

                    if plot_cfg.title:
                        title = str(plot_cfg.title).format(
                            var=var,
                            long_name=long_name,
                            model="ERA5",
                            member="",
                            region=region_label_pretty,
                            season=season,
                            start=start,
                            end=end,
                            baseline_start=str(plot_cfg.baseline.start)[:10],
                            baseline_end=str(plot_cfg.baseline.end)[:10],
                        )
                    else:
                        title = _default_title(
                            long_name=long_name,
                            plev_title=plev_title,
                            proper_model_name="ERA5",
                            member="",
                            region_label=region_label_pretty,
                            season=season,
                            start=start,
                            end=end,
                            plot_cfg=plot_cfg,
                        )

                    ax.set_title(title, pad=10)
                    ax.set_ylabel(_ylabel(long_name, unit_here, plot_cfg))
                    ax.set_xlabel(
                        "Month"
                        if str(plot_cfg.freq).lower() == "monthly"
                        else "Day of year"
                    )

                    if xticks is not None and xticklabels is not None:
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(xticklabels)

                    ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.margins(x=0.01)

                    handles = [
                        Line2D([0], [0], color=col, lw=2.2, label=lab)
                        for lab, col in legend_items
                    ]

                    if bool(plot_cfg.highlight_previous_year) and len(years) > 1:
                        handles.append(
                            Line2D(
                                [0],
                                [0],
                                color="#D55E00",
                                lw=float(plot_cfg.previous_linewidth),
                                label=str(sorted(years)[-2]),
                            )
                        )

                    if bool(plot_cfg.highlight_latest_year) and len(years) > 0:
                        handles.append(
                            Line2D(
                                [0],
                                [0],
                                color="black",
                                lw=float(plot_cfg.latest_linewidth),
                                label=str(max(years)),
                            )
                        )

                    if plot_cfg.legend.inside_plot:
                        ax.legend(
                            handles=handles,
                            loc=plot_cfg.legend.loc,
                            frameon=False,
                            fontsize="small",
                            borderaxespad=0.0,
                        )
                    else:
                        ax.legend(
                            handles=handles,
                            loc="upper center",
                            bbox_to_anchor=(0.5, -0.15),
                            ncol=min(4, max(1, len(handles))),
                            frameon=False,
                            fontsize="small",
                        )

                    if cfg.out.savefig:
                        fig.savefig(outfile, dpi=cfg.out.dpi, bbox_inches="tight")
                        plt.close(fig)
                    else:
                        plt.show()
                        