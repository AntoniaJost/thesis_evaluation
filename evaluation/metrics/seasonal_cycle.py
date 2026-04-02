from __future__ import annotations

import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import hydra

from evaluation.general_functions import (
    ensure_allowed_var,
    iter_vars_and_plevs,
    open_model_da,
    conversion_rules,
    should_compute_output,
    model_abbrev,
    ensemble_mean_as_member,
    plev_strings,
    open_single_match,
    select_plev_if_needed,
    normalise_list,
    format_unit_for_plot
)
from evaluation.metrics.individual_plots import (
    _area_mean,
    _select_bbox,
    _coord_to_dms_tag,
    _wrap_lon_360,
)
from evaluation.metrics.soi import _lat_slice


SEASON_MONTHS = {
    "full": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


def _selected_seasons(plot_cfg) -> list[str]:
    seasons = normalise_list(plot_cfg.season)
    out = []
    for s in seasons:
        s = str(s).strip()
        if s not in SEASON_MONTHS:
            raise ValueError(
                f"plots.seasonal_cycle.season must be one of {list(SEASON_MONTHS.keys())} "
                f"or a list of them. Got: {s}"
            )
        out.append(s)
    return out


def _normalise_region(region) -> str | None:
    if region is None:
        return "global"
    reg = str(region).strip().lower()

    aliases = {
        "global": "global",
        "northern": "northern",
        "nothern": "northern",
        "southern": "southern",
        "tropics": "tropics",
        "arctic": "arctic",
        "artic": "arctic",
        "antarctic": "antarctic",
        "antartic": "antarctic",
        "individual": "individual",
    }
    if reg not in aliases:
        raise ValueError(
            "plots.seasonal_cycle.region must be one of: "
            "global, northern, southern, tropics, arctic, antarctic, individual. "
            f"Got: {region}"
        )
    return aliases[reg]


def _subset_for_region(da: xr.DataArray, plot_cfg) -> xr.DataArray:
    region = _normalise_region(plot_cfg.region)

    if region == "global":
        return da

    if region == "individual":
        lat0 = float(plot_cfg.individual.lat0)
        lat1 = float(plot_cfg.individual.lat1)
        lon0 = float(plot_cfg.individual.lon0)
        lon1 = float(plot_cfg.individual.lon1)
        return _select_bbox(da, lat0, lat1, lon0, lon1)

    if region == "northern":
        return da.sel(lat=_lat_slice(da, 0.0, 90.0))

    if region == "southern":
        return da.sel(lat=_lat_slice(da, -90.0, 0.0))

    if region == "tropics":
        # fixed standard-ish tropical band (nördlicher bis südlicher Wendekreis)
        return da.sel(lat=_lat_slice(da, -23.5, 23.5))

    if region == "arctic":
        min_lat = float(plot_cfg.polar.min_latitude)
        return da.sel(lat=_lat_slice(da, min_lat, 90.0))

    if region == "antarctic":
        max_lat = float(plot_cfg.polar.max_latitude)
        return da.sel(lat=_lat_slice(da, -90.0, max_lat))

    raise ValueError(f"Unsupported region: {region}")


def _region_tag_and_label(plot_cfg) -> tuple[str, str]:
    region = _normalise_region(plot_cfg.region)

    if region != "individual":
        label_map = {
            "global": "Global",
            "northern": "Northern Hemisphere",
            "southern": "Southern Hemisphere",
            "tropics": "Tropics",
            "arctic": "Arctic",
            "antarctic": "Antarctic",
        }
        return region, label_map[region]

    lat0 = float(plot_cfg.individual.lat0)
    lat1 = float(plot_cfg.individual.lat1)
    lon0 = _wrap_lon_360(float(plot_cfg.individual.lon0))
    lon1 = _wrap_lon_360(float(plot_cfg.individual.lon1))

    lat0_tag = _coord_to_dms_tag(lat0, "lat")
    lat1_tag = _coord_to_dms_tag(lat1, "lat")
    lon0_tag = _coord_to_dms_tag(lon0, "lon")
    lon1_tag = _coord_to_dms_tag(lon1, "lon")

    tag = f"box_{lat0_tag}-{lat1_tag}_{lon0_tag}-{lon1_tag}"
    label = f"Box: {lat0_tag}-{lat1_tag}, {lon0_tag}-{lon1_tag}"
    return tag, label


def _open_era5_daily_da(plot_cfg, var: str, start: str, end: str, plev=None) -> xr.DataArray:
    """
    ERA5 daily data loaded from Robert (cmorised), 
    structure: {era5_path}/{var}/{grid}/{var}_day_*_gn_*.nc
    """
    pattern = os.path.join(
        str(plot_cfg.era5_path),
        var,
        str(plot_cfg.grid),
        f"{var}_day_*_gn_*.nc",
    )
    path = open_single_match(pattern)
    ds = xr.open_dataset(path).sel(time=slice(start, end))

    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in {path}. Available: {list(ds.data_vars)}")

    da = ds[var]
    da = select_plev_if_needed(da, var=var, plev=plev, context=path)
    return da


def _seasonal_climatology(series: xr.DataArray, season: str) -> pd.Series:
    """
    builds a day-of-year climatology that preserves leap years
    group by (month, day), so Feb 29 is retained if present

    for DJF, we map Dec to year 2003 and Jan/Feb to 2004 so the season is contiguous (just dummy year)
    for all other seasons/full, we map to leap year 2004
    """
    s = series.to_pandas().dropna()
    months = SEASON_MONTHS[season]
    s = s[s.index.month.isin(months)]

    grouped = s.groupby([s.index.month, s.index.day]).mean()

    dates = []
    values = []

    # dummy plotting to have correct x-axis (2004 = lap year)
    for (month, day), value in grouped.items():
        if season == "DJF":
            year = 2003 if month == 12 else 2004
        else:
            year = 2004
        dates.append(pd.Timestamp(year=year, month=month, day=day))
        values.append(value)

    clim = pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()
    return clim


def _resolve_figsize(plot_cfg):
    if plot_cfg.figsize not in (None, "null"):
        return tuple(plot_cfg.figsize)
    return (11, 5)


def _default_title(long_name: str, plev_title: str, season: str, region_label: str, start: str, end: str) -> str:
    season_label = "Full Year" if season == "full" else season
    return f"Seasonal Cycle of {long_name}{plev_title} | {season_label} | {region_label}\n({start} – {end})"


def _format_title(plot_cfg, var: str, long_name: str, plev, plev_title: str, season: str, region_label: str, start: str, end: str):
    if plot_cfg.title:
        plev_hpa = None if plev is None else int((float(plev) * 100 if float(plev) < 2000 else float(plev)) / 100)
        return str(plot_cfg.title).format(
            var=var,
            long_name=long_name,
            plev=plev,
            plev_hpa=plev_hpa,
            season=season,
            region=region_label,
            start=start,
            end=end,
        )
    return _default_title(long_name, plev_title, season, region_label, start, end)


def _output_filename(var: str, plev_tag: str, model_tags: str, region_tag: str, season: str, start: str, end: str) -> str:
    start_tag = str(start).replace("-", "")
    end_tag = str(end).replace("-", "")
    return f"seasonal_cycle_{var}{plev_tag}_{model_tags}_{region_tag}_{season}_{start_tag}-{end_tag}.png"


def run(cfg):
    plot_cfg = cfg.plots.seasonal_cycle

    if str(plot_cfg.freq).strip().lower() != "daily":
        raise ValueError("seasonal_cycle only supports freq='daily'.")
    if str(plot_cfg.grid).strip().lower() != "gn":
        raise ValueError("seasonal_cycle only supports grid='gn'.")

    seasons = _selected_seasons(plot_cfg)
    figsize = _resolve_figsize(plot_cfg)
    add_dir = str(plot_cfg.special_outdir) if plot_cfg.special_outdir else ""

    for item in iter_vars_and_plevs(cfg, plot_cfg):
        var = item["var"]
        long_name = item["long_name"]
        unit = format_unit_for_plot(item["unit"])
        start = item["start"]
        end = item["end"]

        ensure_allowed_var(cfg, var)

        for plev in item["plevs"]:
            plev_title, plev_tag = plev_strings(plev)

            model_tags = "_".join(model_abbrev(m) for m in plot_cfg.models)
            region_tag, region_label = _region_tag_and_label(plot_cfg)

            outdir = os.path.join(
                hydra.utils.get_original_cwd(),
                cfg.out.dir,
                "seasonal_cycle",
                var,
                add_dir,
            )
            os.makedirs(outdir, exist_ok=True)

            season_to_outfile = {}
            seasons_to_compute = []

            for season in seasons:
                fname = _output_filename(
                    var=var,
                    plev_tag=plev_tag,
                    model_tags=model_tags,
                    region_tag=region_tag,
                    season=season,
                    start=start,
                    end=end,
                )
                outfile = os.path.join(outdir, fname)
                season_to_outfile[season] = outfile

                if not (
                    cfg.out.savefig
                    and not should_compute_output(outfile, getattr(cfg.out, "overwrite", "ask"))
                ):
                    seasons_to_compute.append(season)

            if not seasons_to_compute:
                continue
            
            # ERA5
            era5_da = _open_era5_daily_da(plot_cfg, var=var, start=start, end=end, plev=plev)
            era5_da, unit_here = conversion_rules(var, era5_da, cfg, "era5", unit)
            era5_region = _subset_for_region(era5_da, plot_cfg)
            era5_daily = _area_mean(era5_region)

            # model data
            model_to_member_daily = {}
            for model_name in plot_cfg.models:
                model_cfg = cfg.datasets.models[model_name]
                member_to_daily = {}

                for member in cfg.members:
                    da_model = open_model_da(
                        model_cfg=model_cfg,
                        cfg=cfg,
                        member=member,
                        var=var,
                        modelname=model_cfg.modelname,
                        freq=plot_cfg.freq,
                        start=start,
                        end=end,
                        grid=plot_cfg.grid,
                        plev=plev,
                    )
                    da_model, _ = conversion_rules(var, da_model, cfg, "model", unit_here)
                    da_region = _subset_for_region(da_model, plot_cfg)
                    member_to_daily[member] = _area_mean(da_region)

                if plot_cfg.include_ensemble_mean_as_member:
                    member_to_daily = ensemble_mean_as_member(member_to_daily, name="mean")

                if plot_cfg.only_mean:
                    if not plot_cfg.include_ensemble_mean_as_member:
                        raise ValueError(
                            "plots.seasonal_cycle.only_mean=true requires include_ensemble_mean_as_member=true."
                        )
                    member_to_daily = {"mean": member_to_daily["mean"]}

                model_to_member_daily[model_name] = member_to_daily

            # plot only needed seasons
            for season in seasons_to_compute:
                outfile = season_to_outfile[season]

                era5_clim = _seasonal_climatology(era5_daily, season)

                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

                ax.plot(
                    era5_clim.index,
                    era5_clim.values,
                    color="black",
                    linewidth=1.6,
                    label="ERA5",
                    zorder=5,
                )

                for model_name in plot_cfg.models:
                    proper_model_name = getattr(cfg.datasets.models[model_name], "proper_name", model_name)
                    base_colour = plot_cfg.colours.base_colours[model_name]
                    light_colour = plot_cfg.colours.colours_light[model_name]

                    member_to_daily = model_to_member_daily[model_name]
                    real_members = [m for m in member_to_daily.keys() if m != "mean"]

                    for member in real_members:
                        clim = _seasonal_climatology(member_to_daily[member], season)
                        label = None
                        if not plot_cfg.include_ensemble_mean_as_member:
                            label = f"{proper_model_name} {member}"
                        ax.plot(
                            clim.index,
                            clim.values,
                            color=light_colour,
                            linewidth=0.9,
                            alpha=0.9,
                            label=label,
                            zorder=2,
                        )

                    if plot_cfg.include_ensemble_mean_as_member and "mean" in member_to_daily:
                        clim_mean = _seasonal_climatology(member_to_daily["mean"], season)
                        ax.plot(
                            clim_mean.index,
                            clim_mean.values,
                            color=base_colour,
                            linewidth=1.6,
                            alpha=1.0,
                            label=f"{proper_model_name} mean",
                            zorder=4,
                        )

                title = _format_title(
                    plot_cfg=plot_cfg,
                    var=var,
                    long_name=long_name,
                    plev=plev,
                    plev_title=plev_title,
                    season=season,
                    region_label=region_label,
                    start=start,
                    end=end,
                )

                ax.set_title(title, pad=10)
                ax.set_ylabel(f"{long_name} ({unit_here})")
                ax.set_xlabel("Month")

                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

                if season == "full":
                    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
                else:
                    ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=[8, 15, 22]))

                ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
                ax.grid(True, which="minor", linewidth=0.5, alpha=0.2)

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.margins(x=0.01)

                if plot_cfg.legend.inside_plot:
                    ax.legend(loc=plot_cfg.legend.loc, frameon=False, fontsize="small", borderaxespad=0.0)
                else:
                    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize="small")

                if cfg.out.savefig:
                    fig.savefig(outfile, dpi=cfg.out.dpi, bbox_inches="tight")
                    plt.close(fig)
                else:
                    plt.show()
                    