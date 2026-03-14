from __future__ import annotations

import math
import os
# from functools import lru_cache

import cartopy.crs as ccrs
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point

from evaluation.general_functions import (
    get_range_from_csv,
    conversion_rules,
    ensemble_mean_as_member,
    ensure_allowed_var,
    iter_vars_and_plevs,
    model_abbrev,
    open_era5_da,
    open_model_da,
    plev_strings,
    should_compute_output,
)
from evaluation.metrics.global_mean import(annual_weighted_mean, lin_reg, trend_decay)
from evaluation.metrics.anomalies import(to_anomaly)
from evaluation.metrics.bias_map import(compute_slope_per_gridpoint)
from evaluation.metrics.soi import (_lat_slice)



POLAR_LOCATIONS = {"arctic", "antarctic"}


def _normalise_location(location) -> str | None:
    if location is None:
        return None
    loc = str(location).strip().lower()
    if loc == "global":
        return None
    if loc == "artic": # stupidity catch for Antonias who cannot spell "arctic" correctly
        loc = "arctic"
    if loc == "antartic":
        loc = "antarctic"
    allowed = {None, "individual", "arctic", "antarctic"}
    if loc not in allowed:
        raise ValueError(
            "plots.individual_plots.location must be one of: null/global, individual, arctic, antarctic. "
            f"Got: {location}"
        )
    return loc


def _time_stat(plot_cfg) -> str:
    stat = str(getattr(plot_cfg, "time_stat", "raw")).strip().lower()
    allowed = {"raw", "annual_mean", "trend"}
    if stat not in allowed:
        raise ValueError(
            f"plots.individual_plots.time_stat must be one of {sorted(allowed)}. Got: {stat}"
        )
    return stat


def _validate_time_order(start: str, end: str):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError(f"Invalid time range: end ({end}) must not be earlier than start ({start}).")


def _as_float_or_none(value):
    if value is None:
        return None
    return float(value)


def _wrap_lon_360(lon: float) -> float:
    # converts values from [-180,180] to 0...360
    return float(lon) % 360.0


# def _lat_slice(da: xr.DataArray, lat0: float, lat1: float):
#     lat = da["lat"]
#     if float(lat[0]) < float(lat[-1]):
#         return slice(min(lat0, lat1), max(lat0, lat1))
#     return slice(max(lat0, lat1), min(lat0, lat1))

def _format_time_from_freq(ts, freq: str) -> str:
    # formats time correctly, depending on frequence from config
    ts = pd.Timestamp(ts)
    if freq == "monthly":
        return ts.strftime("%Y-%m")
    if freq == "daily":
        return ts.strftime("%Y-%m-%d")
    raise ValueError(f"Unsupported frequence: {freq}. Expected 'monthly' or 'daily'.")
 

def _nearest_time_str(da: xr.DataArray, requested: str, freq: str) -> str:
    # gets the nearest timestep
    req = np.datetime64(requested)
    idx = int(np.argmin(np.abs(da.time.values - req)))
    return _format_time_from_freq(da.time.values[idx], freq)
# str(pd.Timestamp(da.time.values[idx]).date())

def _time_label(start: str, end: str, method: str, da: xr.DataArray, freq: str, plot_cfg) -> str:
    stat = _time_stat(plot_cfg)
    if stat in {"annual_mean", "trend"}:
        start_y = pd.Timestamp(start).year
        end_y = pd.Timestamp(end).year
        return f"{start_y} to {end_y}"
    # prepares the correct time label, if freq=monthly: yyyy-mm, if freq=daily: yyyy-mm-dd
    if pd.Timestamp(start) == pd.Timestamp(end):
        return _nearest_time_str(da, start, freq)
    start_str = _format_time_from_freq(start, freq)
    end_str = _format_time_from_freq(end, freq)
    if method == "map":
        return f"{start_str} to {end_str} (time mean)"
    return f"{start_str} to {end_str}"


# @lru_cache
# def _load_range_table(path: str) -> pd.DataFrame:
#     return pd.read_csv(path)


def _summary_column_prefix(single_time: bool) -> str:
    return "raw" if single_time else "temporal"


# def _summary_bounds_from_csv(csv_path: str, var: str, plev, percentile, prefix: str) -> tuple[float, float]:
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(
#             f"Configured range summary CSV not found: {csv_path}. "
#             "Either create it first or disable CSV-based colour ranges by using anomaly=true."
#         )

#     df = _load_range_table(csv_path)
#     var_col = "variable" if "variable" in df.columns else "var"
#     df = df[df[var_col] == var]

#     if plev is not None:
#         if "plev_pa" not in df.columns:
#             raise ValueError(f"CSV file {csv_path} does not contain a 'plev_pa' column.")
#         plev_pa = accept_Pa_and_hPa(plev, df["plev_pa"].dropna().values)
#         df = df[np.isclose(df["plev_pa"].fillna(-9999.0), plev_pa)]
#     else:
#         if "plev_pa" in df.columns:
#             df = df[df["plev_pa"].isna()]

#     if df.empty:
#         raise ValueError(f"No range information found in {csv_path} for var={var}, plev={plev}.")

#     row = df.iloc[0]
#     perc = str(percentile).lower()
#     if perc == "99":
#         return float(row[f"{prefix}_p01"]), float(row[f"{prefix}_p99"])
#     if perc == "95":
#         return float(row[f"{prefix}_p05"]), float(row[f"{prefix}_p95"])
#     if perc == "raw":
#         return float(row[f"{prefix}_min"]), float(row[f"{prefix}_max"])
#     raise ValueError(f"Unknown range_source.percentile value: {percentile}")


def _dynamic_bounds(arrays: list[xr.DataArray], percentile=99, symmetric=False) -> tuple[float, float]:
    vals = []
    for da in arrays:
        data = np.asarray(da.values).ravel()
        data = data[np.isfinite(data)]
        if data.size:
            vals.append(data)
    if not vals:
        raise ValueError("Could not determine plotting range because all candidate arrays are empty or NaN.")
    vals = np.concatenate(vals)

    perc = str(percentile).lower()
    if perc == "99":
        vmin, vmax = np.nanpercentile(vals, [1, 99])
    elif perc == "95":
        vmin, vmax = np.nanpercentile(vals, [5, 95])
    elif perc == "raw":
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    else:
        raise ValueError(f"Unknown percentile option: {percentile}")

    if symmetric:
        vmax_abs = max(abs(float(vmin)), abs(float(vmax)))
        if vmax_abs == 0:
            vmax_abs = 1e-12
        return -vmax_abs, vmax_abs

    if float(vmin) == float(vmax):
        eps = max(abs(float(vmin)) * 0.05, 1e-12)
        return float(vmin) - eps, float(vmax) + eps
    return float(vmin), float(vmax)


def _prepare_member_mapping(cfg, plot_cfg, member_to_da: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    if plot_cfg.include_ensemble_mean_as_member:
        member_to_da = ensemble_mean_as_member(member_to_da, name="mean")

    if plot_cfg.only_mean:
        if not plot_cfg.include_ensemble_mean_as_member:
            raise ValueError(
                "plots.individual_plots.only_mean=true requires include_ensemble_mean_as_member=true."
            )
        if "mean" not in member_to_da:
            raise ValueError("Requested only_mean=true, but ensemble mean could not be constructed.")
        return {"mean": member_to_da["mean"]}

    out = {m: member_to_da[m] for m in cfg.members}
    if plot_cfg.include_ensemble_mean_as_member:
        out["mean"] = member_to_da["mean"]
    return out


def _select_bbox(da: xr.DataArray, lat0: float, lat1: float, lon0: float, lon1: float) -> xr.DataArray:
    lat0 = float(lat0)
    lat1 = float(lat1)
    lon0 = _wrap_lon_360(lon0)
    lon1 = _wrap_lon_360(lon1)

    lat_sel = da.sel(lat=_lat_slice(da, lat0, lat1))

    if lat_sel.sizes.get("lat", 0) == 0:
        raise ValueError(
            f"Selected latitude range is empty: lat0={lat0}, lat1={lat1}. "
            "Please increase the distance between lat0 and lat1 for a proper map."
        )
    # longitude selection, using "nearest" method
    if math.isclose(lon0, lon1):
        lon_sel = lat_sel.sel(lon=lon0, method="nearest")
        if "lon" not in lon_sel.dims:
            lon_val = float(lon_sel["lon"].values)
            lon_sel = lon_sel.expand_dims(lon=[lon_val])
        if lon_sel.sizes.get("lon", 0) == 0:
            raise ValueError(
                f"Selected longitude range is empty: lon0={lon0}, lon1={lon1}. "
                "Please increase the distance between lon0 and lon1 for a proper map."
            )
        return lon_sel

    if lon0 < lon1:
        return lat_sel.sel(lon=slice(lon0, lon1))
    else:
        part1 = lat_sel.sel(lon=slice(lon0, 360))
        part2 = lat_sel.sel(lon=slice(0, lon1))
        out = xr.concat([part1, part2], dim="lon")
        _, unique_idx = np.unique(out["lon"].values, return_index=True)
        out = out.isel(lon=np.sort(unique_idx))

    if out.sizes.get("lon", 0) == 0 or out.sizes.get("lat", 0) == 0:
        raise ValueError(
            f"Selected bounding box contains no grid cells: "
            f"lat0={lat0}, lat1={lat1}, lon0={lon0}, lon1={lon1}."
        )

    return out


def subset_for_location(da: xr.DataArray, plot_cfg) -> xr.DataArray:
    location = _normalise_location(plot_cfg.location)

    if location is None:
        return da

    if location == "individual":
        lat0 = _as_float_or_none(plot_cfg.individual.lat0)
        lat1 = _as_float_or_none(plot_cfg.individual.lat1)
        lon0 = _as_float_or_none(plot_cfg.individual.lon0)
        lon1 = _as_float_or_none(plot_cfg.individual.lon1)
        if None in (lat0, lat1, lon0, lon1):
            raise ValueError(
                "For location='individual', plots.individual_plots.individual.lat0/lat1/lon0/lon1 must all be set."
            )
        if not (-90 <= lat0 <= 90 and -90 <= lat1 <= 90):
            raise ValueError("Latitude bounds must lie within [-90, 90].")
        return _select_bbox(da, lat0, lat1, lon0, lon1)

    if location == "arctic":
        min_lat = float(plot_cfg.polar.min_latitude)
        if not (-90 < min_lat < 90):
            raise ValueError("plots.individual_plots.polar.min_latitude must lie between -90 and 90.")
        return da.sel(lat=_lat_slice(da, min_lat, 90.0))

    if location == "antarctic":
        max_lat = float(plot_cfg.polar.max_latitude)
        if not (-90 < max_lat < 90):
            raise ValueError("plots.individual_plots.polar.max_latitude must lie between -90 and 90.")
        return da.sel(lat=_lat_slice(da, -90.0, max_lat))

    raise ValueError(f"Unsupported location mode: {location}")


def area_mean(da: xr.DataArray) -> xr.DataArray:
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"Expected 'lat' and 'lon' dimensions, got {da.dims}")
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights).mean(dim=("lat", "lon"))


def baseline_mean(da: xr.DataArray, baseline_start: str, baseline_end: str) -> xr.DataArray:
    base = da.sel(time=slice(baseline_start, baseline_end))
    if base.sizes.get("time", 0) == 0:
        raise ValueError(
            f"Baseline period {baseline_start} to {baseline_end} has no overlap with the selected data."
        )
    return base.mean(dim="time")


def apply_anomaly(da: xr.DataArray, plot_cfg) -> xr.DataArray:
    return da - baseline_mean(da, plot_cfg.baseline.start, plot_cfg.baseline.end)


# def prepare_field(da: xr.DataArray, plot_cfg, method: str) -> xr.DataArray:
#     da_loc = subset_for_location(da, plot_cfg)
#     if plot_cfg.anomaly:
#         da_loc, _ = to_anomaly(da_loc, plot_cfg.baseline.start, plot_cfg.baseline.end) #apply_anomaly(da_loc, plot_cfg)
#     if method == "map":
#         if da_loc.sizes.get("time", 0) == 0:
#             raise ValueError("No timesteps remain after time selection.")
#         return da_loc.mean(dim="time")
#     if method == "timeseries":
#         return area_mean(da_loc)
#     raise ValueError(f"Unsupported method: {method}")

def prepare_field(da: xr.DataArray, plot_cfg, method: str) -> xr.DataArray:
    stat = _time_stat(plot_cfg)
    da_loc = subset_for_location(da, plot_cfg)
    if plot_cfg.anomaly:
        da_loc, _ = to_anomaly(da_loc, plot_cfg.baseline.start, plot_cfg.baseline.end)

    if stat == "raw":
        if method == "map":
            if da_loc.sizes.get("time", 0) == 0:
                raise ValueError("No timesteps remain after time selection.")
            return da_loc.mean(dim="time")
        if method == "timeseries":
            return area_mean(da_loc)

    elif stat == "annual_mean":
        if method == "map":
            da_ann = annual_weighted_mean(da_loc)
            if da_ann.sizes.get("time", 0) == 0:
                raise ValueError("No annual timesteps remain after aggregation.")
            return da_ann.mean(dim="time")

        if method == "timeseries":
            return annual_weighted_mean(area_mean(da_loc))

    elif stat == "trend":
        if method == "map":
            if da_loc.sizes.get("time", 0) < 2:
                raise ValueError("Need at least 2 timesteps to compute a trend map.")
            return compute_slope_per_gridpoint(da_loc) * 10.0

        if method == "timeseries":
            # same data basis as global_mean/anomalies:
            # annual mean series + trend line in plotting
            return annual_weighted_mean(area_mean(da_loc))

    raise ValueError(f"Unsupported method/stat combination: method={method}, time_stat={stat}")


def _projection_and_extent(plot_cfg):
    location = _normalise_location(plot_cfg.location)
    if location is None:
        return ccrs.Robinson(), None
    if location == "individual":
        lon0 = _wrap_lon_360(float(plot_cfg.individual.lon0))
        lon1 = _wrap_lon_360(float(plot_cfg.individual.lon1))
        if math.isclose(lon0, lon1):
            pad = float(getattr(plot_cfg.individual, "point_pad_deg", 2.0))
            extent = [lon0 - pad, lon0 + pad, float(plot_cfg.individual.lat0) - pad, float(plot_cfg.individual.lat1) + pad]
        else:
            lon_min = lon0
            lon_max = lon1
            if lon0 > lon1:
                lon_max += 360.0
            extent = [lon_min, lon_max, min(float(plot_cfg.individual.lat0), float(plot_cfg.individual.lat1)), max(float(plot_cfg.individual.lat0), float(plot_cfg.individual.lat1))]
        return ccrs.PlateCarree(), extent
    if location == "arctic":
        return ccrs.NorthPolarStereo(), [-180, 180, float(plot_cfg.polar.min_latitude), 90]
    if location == "antarctic":
        return ccrs.SouthPolarStereo(), [-180, 180, -90, float(plot_cfg.polar.max_latitude)]
    return ccrs.Robinson(), None


def _plot_single_map(ax, da: xr.DataArray, title: str, cfg, plot_cfg, vmin: float, vmax: float):
    projection, extent = _projection_and_extent(plot_cfg)
    location = _normalise_location(plot_cfg.location)

    ax.set_title(title, fontsize=10)
    ax.coastlines(linewidth=0.9, color=str(plot_cfg.coastline_colour))

    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.gridlines(draw_labels=True, linewidth=0.5, color="black", alpha=0.35, linestyle="--")

    # for global maps only: add cyclic point to avoid seam at 0/360
    if location is None or location in POLAR_LOCATIONS:
        data_cyc, lon_cyc = add_cyclic_point(da.values, coord=da["lon"].values)
        cf = ax.contourf(
            lon_cyc,
            da["lat"].values,
            data_cyc,
            levels=np.linspace(vmin, vmax, 21),
            cmap=str(plot_cfg.colour_scheme),
            extend="both",
            transform=ccrs.PlateCarree(),
        )
    else:
        if da.sizes.get("lat", 0) == 0 or da.sizes.get("lon", 0) == 0:
            raise ValueError(
                "Cannot plot map because the selected region contains no grid cells."
            )
        if da.sizes.get("lat", 0) < 2 or da.sizes.get("lon", 0) < 2:
            raise ValueError(
                "Selected region is too small for map plotting. "
                "For maps, please choose a bounding box that contains at least 2 latitudes and 2 longitudes."
            )
        cf = ax.contourf(
            da["lon"].values,
            da["lat"].values,
            da.values,
            levels=np.linspace(vmin, vmax, 21),
            cmap=str(plot_cfg.colour_scheme),
            extend="both",
            transform=ccrs.PlateCarree(),
        )
    return cf


def _plot_timeseries(ax, era5_series: xr.DataArray, model_series: xr.DataArray, plot_cfg,
                    model_name: str, proper_model_name: str, member: str,  unit_here: str):
    colours = plot_cfg.colours
    base_colour = colours.base_colours[model_name]
    light_colour = colours.colours_light[model_name]
    stat = _time_stat(plot_cfg)
    if stat == "trend":
        # ERA5 only shown if not plotting difference
        if not plot_cfg.difference:
            lrg_era5, slope_era5 = lin_reg(era5_series)
            ax.plot(
                era5_series["time"].dt.year.values,
                era5_series.values,
                color="black",
                linewidth=1.4,
                label=f"ERA5 (Trend: {trend_decay(slope_era5)} {unit_here}/decade)",
            )
            ax.plot(
                era5_series["time"].dt.year.values,
                lrg_era5,
                color="black",
                linewidth=1.0,
                linestyle="--",
                alpha=0.9,
            )

        lrg_model, slope_model = lin_reg(model_series)
        ax.plot(
            model_series["time"].dt.year.values,
            model_series.values,
            color=base_colour if member == "mean" else light_colour,
            linewidth=1.6 if member == "mean" else 1.0,
            alpha=1.0 if member == "mean" else 0.95,
            label=f"{proper_model_name} {member} (Trend: {trend_decay(slope_model)} {unit_here}/decade)",
        )
        ax.plot(
            model_series["time"].dt.year.values,
            lrg_model,
            color=base_colour if member == "mean" else light_colour,
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
        )

        if plot_cfg.difference:
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)

    else:
        if plot_cfg.difference:
            ax.plot(
                model_series["time"].values,
                model_series.values,
                color=base_colour,
                linewidth=1.5,
                label=f"{proper_model_name} {member} - ERA5",
            )
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
        else:
            ax.plot(
                era5_series["time"].values,
                era5_series.values,
                color="black",
                linewidth=1.4,
                label="ERA5",
            )
            ax.plot(
                model_series["time"].values,
                model_series.values,
                color=base_colour if member == "mean" else light_colour,
                linewidth=1.6 if member == "mean" else 1.0,
                alpha=1.0 if member == "mean" else 0.95,
                label=f"{proper_model_name} {member}",
            )

    ax.set_xlabel("Year" if stat in {"annual_mean", "trend"} else (str(plot_cfg.xlabel) if plot_cfg.xlabel is not None else "Time"))
    # if plot_cfg.difference:
    #     ax.plot(
    #         model_series["time"].values,
    #         model_series.values,
    #         color=base_colour,
    #         linewidth=1.5,
    #         label=f"{proper_model_name} {member}",
    #     )
    #     ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    # else:
    #     ax.plot(
    #         era5_series["time"].values,
    #         era5_series.values,
    #         color="black",
    #         linewidth=1.4,
    #         label="ERA5",
    #     )
    #     ax.plot(
    #         model_series["time"].values,
    #         model_series.values,
    #         color=base_colour if member == "mean" else light_colour,
    #         linewidth=1.6 if member == "mean" else 1.0,
    #         alpha=1.0 if member == "mean" else 0.95,
    #         label=f"{proper_model_name} {member}",
    #     )

    # ax.set_xlabel(str(plot_cfg.xlabel) if plot_cfg.xlabel is not None else "Time")
    if plot_cfg.ylabel is not None:
        ax.set_ylabel(str(plot_cfg.ylabel))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc=str(plot_cfg.legend_loc), frameon=False)


def _canonicalise_time_for_difference(da: xr.DataArray, plot_cfg) -> xr.DataArray:
    stat = _time_stat(plot_cfg)
    freq = str(plot_cfg.freq).strip().lower()

    if "time" not in da.dims:
        return da

    # annual_mean and trend: align by year only
    if stat in {"annual_mean", "trend"}:
        years = da["time"].dt.year.values
        return da.assign_coords(time=years)

    # raw monthly: align by year-month
    if stat == "raw" and freq == "monthly":
        ym = pd.to_datetime(da["time"].dt.strftime("%Y-%m-01").values)
        return da.assign_coords(time=ym)

    # raw daily: align by date only
    if stat == "raw" and freq == "daily":
        days = pd.to_datetime(da["time"].dt.strftime("%Y-%m-%d").values)
        return da.assign_coords(time=days)

    return da


def _subtract_with_time_alignment(model_da: xr.DataArray, era5_da: xr.DataArray, plot_cfg) -> xr.DataArray:
    if "time" in model_da.dims and "time" in era5_da.dims:
        model_da = _canonicalise_time_for_difference(model_da, plot_cfg)
        era5_da = _canonicalise_time_for_difference(era5_da, plot_cfg)

        model_da, era5_da = xr.align(model_da, era5_da, join="inner")

        if model_da.sizes.get("time", 0) == 0:
            raise ValueError(
                "No overlapping timesteps remain after aligning model and ERA5 for difference plot. "
                f"time_stat={_time_stat(plot_cfg)}, freq={plot_cfg.freq}"
            )

    return model_da - era5_da


def _resolve_figsize(plot_cfg, method: str):
    """
    determines the figure size
    order:
    1. if figsize is set in config: use it
    2. if map + polar: [8, 8]
    3. if map + global: [9, 5]
    4. otherwise: let matplotlib decide (None)
    """
    if plot_cfg.figsize not in (None, "null"):
        return tuple(plot_cfg.figsize)
    if method == "map":
        loc = _normalise_location(plot_cfg.location)
        if loc in POLAR_LOCATIONS:
            return (8, 8)
        if loc is None:  # global
            return (9, 5)
    if method == "timeseries":
        return (12, 4)
    return None


def _default_title(plot_cfg, method: str, long_name: str, proper_model_name: str, member: str, time_label: str, plev_title: str):
    stat = _time_stat(plot_cfg)
    # title construction if no title is passed in the config
    if method == "map":
        if stat == "trend":
            lead = "Decadal trend difference to ERA5" if plot_cfg.difference else f"{proper_model_name} decadal trend"
        elif stat == "annual_mean":
            lead = "Annual mean difference to ERA5" if plot_cfg.difference else f"{proper_model_name} annual mean"
        else:
            lead = "Difference to ERA5" if plot_cfg.difference else proper_model_name

        if plot_cfg.anomaly and plot_cfg.difference:
            lead += " (anomaly)"
        elif plot_cfg.anomaly and not plot_cfg.difference:
            lead += " anomaly"

        title = f"{lead}: {long_name}{plev_title} | {member} | {time_label}"

    else:
        if stat == "annual_mean" or stat == "trend":
            lead = "Annual mean timeseries"
            if plot_cfg.difference:
                lead = "Annual mean difference timeseries"
        else:
            lead = "Area-mean difference to ERA5" if plot_cfg.difference else "Area-mean timeseries"

        if plot_cfg.anomaly and plot_cfg.difference:
            lead += " (anomaly)"
        elif plot_cfg.anomaly and not plot_cfg.difference:
            lead += " anomaly"

        title = f"{lead}: {long_name}{plev_title} | {proper_model_name} {member} | {time_label}"
    # if method == "map":
    #     lead = "Difference to ERA5" if plot_cfg.difference else proper_model_name
    #     if plot_cfg.anomaly and plot_cfg.difference:
    #         lead = "Difference to ERA5 (anomaly)"
    #     elif plot_cfg.anomaly:
    #         lead = f"{proper_model_name} anomaly"
    #     title = f"{lead}: {long_name}{plev_title} | {member} | {time_label}"
    # else: 
    #     lead = "Area-mean difference to ERA5" if plot_cfg.difference else "Area-mean timeseries"
    #     if plot_cfg.anomaly and plot_cfg.difference:
    #         lead = "Area-mean anomaly difference to ERA5"
    #     elif plot_cfg.anomaly:
    #         lead = "Area-mean anomaly timeseries"
    #     title = f"{lead}: {long_name}{plev_title} | {proper_model_name} {member} | {time_label}"
    # add baseline information as second line if anomaly is used
    if plot_cfg.anomaly:
        base_start = _format_time_from_freq(plot_cfg.baseline.start, plot_cfg.freq)
        base_end = _format_time_from_freq(plot_cfg.baseline.end, plot_cfg.freq)
        title += f"\nBaseline removed: mean of {base_start} – {base_end}"

    return title

def _format_title(plot_cfg, method: str, var: str, long_name: str, model_name: str, proper_model_name: str, member: str, plev, plev_title: str, start: str, end: str, time_label: str):
    if plot_cfg.title:
        plev_hpa = None if plev is None else int((float(plev) * 100 if float(plev) < 2000 else float(plev)) / 100)
        return str(plot_cfg.title).format(
            var=var,
            long_name=long_name,
            model=model_name,
            proper_model_name=proper_model_name,
            member=member,
            plev=plev,
            plev_hpa=plev_hpa,
            method=method,
            start=start,
            end=end,
            time_label=time_label,
        )
    return _default_title(plot_cfg, method, long_name, proper_model_name, member, time_label, plev_title)


def _coord_to_dms_tag(coord: float, axis: str) -> str:
    """
    converts lat/lon config setting into suitable filenames
    e.g.: 50.234923 → 50-14-05N, -12.5 → 12-30-00S
    """
    coord = float(coord)
    sign = 1 if coord >= 0 else -1
    coord_abs = abs(coord)
    deg = int(coord_abs)
    minutes_full = (coord_abs - deg) * 60
    minutes = int(minutes_full)
    seconds = int(round((minutes_full - minutes) * 60))
    # fix rounding overflow like 59.9999 → 60
    if seconds == 60:
        seconds = 0
        minutes += 1
    if minutes == 60:
        minutes = 0
        deg += 1
    if axis == "lat":
        hemi = "N" if sign >= 0 else "S"
    elif axis == "lon":
        hemi = "E" if sign >= 0 else "W"
    else:
        raise ValueError("axis must be 'lat' or 'lon'")
    return f"{deg}-{minutes:02d}-{seconds:02d}{hemi}"


def _output_filename(method: str, var: str, plev_tag: str, model_name: str, member: str, 
                     start: str, end: str, plot_cfg) -> str:
    model_tag = model_abbrev(model_name)
    stat = _time_stat(plot_cfg)
    if stat in {"annual_mean", "trend"}:
        start_tag = str(pd.Timestamp(start).year)
        end_tag = str(pd.Timestamp(end).year)
    else:
        start_str = _format_time_from_freq(start, plot_cfg.freq)
        end_str = _format_time_from_freq(end, plot_cfg.freq)
        start_tag = str(start_str).replace("-", "")
        end_tag = str(end_str).replace("-", "")
    # start_str = _format_time_from_freq(start, plot_cfg.freq)
    # end_str = _format_time_from_freq(end, plot_cfg.freq)
    # start_tag = str(start_str).replace("-", "")
    # end_tag = str(end_str).replace("-", "")
    loc = _normalise_location(plot_cfg.location)
    loc_tag = "global" if loc is None else loc
    if loc == "individual":
        lat0 = float(plot_cfg.individual.lat0)
        lat1 = float(plot_cfg.individual.lat1)
        lon0 = _wrap_lon_360(float(plot_cfg.individual.lon0))
        lon1 = _wrap_lon_360(float(plot_cfg.individual.lon1))
        lat0_tag = _coord_to_dms_tag(lat0, "lat")
        lat1_tag = _coord_to_dms_tag(lat1, "lat")
        lon0_tag = _coord_to_dms_tag(lon0, "lon")
        lon1_tag = _coord_to_dms_tag(lon1, "lon")
        loc_tag = f"box_{lat0_tag}-{lat1_tag}_{lon0_tag}-{lon1_tag}"
    diff_tag = "_minusERA5" if plot_cfg.difference else ""
    anom_tag = "_anom" if plot_cfg.anomaly else ""
    stat_tag = "_decadalTrend" if stat == "trend" else ""
    return f"{method}_{var}{plev_tag}_{model_tag}_{member}_{loc_tag}{diff_tag}{anom_tag}_{start_tag}-{end_tag}{stat_tag}.png"


def _get_map_bounds(cfg, plot_cfg, arrays: list[xr.DataArray], var: str, plev, difference: bool, anomaly: bool, single_time: bool):
    use_csv = not anomaly
    if use_csv:
        try:
            csv_rel = plot_cfg.range_source.csv_file2 if difference else plot_cfg.range_source.csv_file1
            csv_path = os.path.join(hydra.utils.get_original_cwd(), csv_rel)
            prefix = _summary_column_prefix(single_time)
            vmin, vmax = get_range_from_csv(
                percentile=plot_cfg.range_source.percentile,
                csv_file=csv_path,
                var=var,
                plev=plev,
                prefix=prefix,
            )
            if difference:
                vmax_abs = max(abs(vmin), abs(vmax))
                return -vmax_abs, vmax_abs
            return vmin, vmax
        except Exception as exc:
            print(f"Falling back to dynamic colour range for {var}, plev={plev}: {exc}")

    return _dynamic_bounds(
        arrays,
        percentile=plot_cfg.range_source.percentile,
        symmetric=difference or anomaly,
    )


def run(cfg):
    plot_cfg = cfg.plots.individual_plots
    method = str(plot_cfg.method).strip().lower() if plot_cfg.method is not None else None
    if method not in {"timeseries", "map"}:
        raise ValueError("plots.individual_plots.method must be either 'timeseries' or 'map'.")
    figsize = _resolve_figsize(plot_cfg, method)

    for item in iter_vars_and_plevs(cfg, plot_cfg):
        var = item["var"]
        long_name = item["long_name"]
        unit = item["unit"]
        start = item["start"]
        end = item["end"]
        _validate_time_order(start, end)
        ensure_allowed_var(cfg, var)

        for plev in item["plevs"]:
            plev_title, plev_tag = plev_strings(plev)

            era5 = open_era5_da(cfg, var=var, start=start, end=end, plev=plev)
            era5, unit_here = conversion_rules(var, era5, cfg, "era5", unit)
            era5_prepared = prepare_field(era5, plot_cfg, method)
            time_label = _time_label(start, end, method, era5, plot_cfg.freq, plot_cfg)
            single_time = pd.Timestamp(start) == pd.Timestamp(end)

            for model_name in plot_cfg.models:
                model_cfg = cfg.datasets.models[model_name]
                proper_model_name = getattr(model_cfg, "proper_name", model_name)

                member_to_da = {}
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
                    prepared = prepare_field(da_model, plot_cfg, method)
                    if plot_cfg.difference:
                        prepared = _subtract_with_time_alignment(prepared, era5_prepared, plot_cfg)
                    member_to_da[member] = prepared

                member_to_plot = _prepare_member_mapping(cfg, plot_cfg, member_to_da)

                outdir = os.path.join(
                    hydra.utils.get_original_cwd(),
                    cfg.out.dir,
                    "individual_plots",
                    method,
                    var,
                )
                os.makedirs(outdir, exist_ok=True)

                if method == "timeseries":
                    era5_series = era5_prepared
                    for member, series in member_to_plot.items():
                        if "time" not in series.dims:
                            raise ValueError(
                                f"Expected a time dimension for timeseries plotting, got {series.dims} for {var}."
                            )
                        fname = _output_filename(method, var, plev_tag, model_name, member, start, end, plot_cfg)
                        outfile = os.path.join(outdir, fname)
                        if cfg.out.savefig and not should_compute_output(outfile, getattr(cfg.out, "overwrite", "ask")):
                            continue

                        fig, ax = plt.subplots(figsize=figsize)
                        _plot_timeseries(ax, era5_series, series, plot_cfg, model_name, proper_model_name, member, unit_here)
                        title = _format_title(
                            plot_cfg,
                            method,
                            var,
                            long_name,
                            model_name,
                            proper_model_name,
                            member,
                            plev,
                            plev_title,
                            start,
                            end,
                            time_label,
                        )
                        ax.set_title(title)
                        ylabel = str(plot_cfg.ylabel) if plot_cfg.ylabel is not None else unit_here
                        if plot_cfg.difference and ylabel == unit_here:
                            ylabel = f"Difference ({unit_here})"
                        elif plot_cfg.anomaly and ylabel == unit_here:
                            ylabel = f"Anomaly in {long_name} ({unit_here})"
                        elif ylabel == unit_here:
                            ylabel = f"{long_name} ({unit_here})"
                        ax.set_ylabel(ylabel)

                        if cfg.out.savefig:
                            fig.savefig(outfile, dpi=cfg.out.dpi, bbox_inches="tight")
                            plt.close(fig)
                        else:
                            plt.show()
                else:
                    arrays = list(member_to_plot.values())
                    vmin, vmax = _get_map_bounds(
                        cfg,
                        plot_cfg,
                        arrays,
                        var,
                        plev,
                        difference=bool(plot_cfg.difference),
                        anomaly=bool(plot_cfg.anomaly),
                        single_time=single_time,
                    )
                    projection, _ = _projection_and_extent(plot_cfg)

                    for member, map_da in member_to_plot.items():
                        fname = _output_filename(method, var, plev_tag, model_name, member, start, end, plot_cfg)
                        outfile = os.path.join(outdir, fname)
                        if cfg.out.savefig and not should_compute_output(outfile, getattr(cfg.out, "overwrite", "ask")):
                            continue

                        fig, ax = plt.subplots(
                            figsize=figsize,
                            subplot_kw={"projection": projection},
                        )
                        fig.subplots_adjust(top=0.9, bottom=0.12)
                        cf = _plot_single_map(ax, map_da, "", cfg, plot_cfg, vmin=vmin, vmax=vmax)
                        title = _format_title(
                            plot_cfg,
                            method,
                            var,
                            long_name,
                            model_name,
                            proper_model_name,
                            member,
                            plev,
                            plev_title,
                            start,
                            end,
                            time_label,
                        )
                        ax.set_title(title, pad=13)
                        cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.1, shrink=0.9)
                        cbar_label = unit_here
                        if _time_stat(plot_cfg) == "trend":
                            if plot_cfg.difference:
                                cbar_label = f"Model - ERA5 ({unit_here}/decade)"
                            else:
                                cbar_label = f"{long_name} ({unit_here}/decade)"
                        elif plot_cfg.difference:
                            cbar_label = f"Model - ERA5 ({unit_here})"
                        elif plot_cfg.anomaly:
                            cbar_label = f"Anomaly ({unit_here})"
                        else:
                            cbar_label = f"{long_name} ({unit_here})"
                        cbar.set_label(cbar_label)

                        if cfg.out.savefig:
                            fig.savefig(outfile, dpi=cfg.out.dpi, bbox_inches="tight")
                            plt.close(fig)
                        else:
                            plt.show()
