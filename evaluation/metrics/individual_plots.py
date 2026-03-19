# evaluation/metrics/individual_plots.py
from __future__ import annotations

import math
import os

import cartopy.crs as ccrs
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
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
    detrend_dataarray,
)
from evaluation.metrics.global_mean import(annual_weighted_mean, lin_reg, trend_decay)
from evaluation.metrics.anomalies import(to_anomaly)
from evaluation.metrics.bias_map import(compute_slope_per_gridpoint, nice_bin_size, build_zero_bin_levels, symmetric_ticks_from_levels)
from evaluation.metrics.soi import (_lat_slice)


POLAR_LOCATIONS = {"arctic", "antarctic"}

# ---- CONFIG / VALIDATION HELPERS ----
def _normalise_location(location) -> str | None:
    # get location
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
    # gets time frequency
    stat = str(getattr(plot_cfg, "time_stat", "raw")).strip().lower()
    allowed = {"raw", "annual_mean", "trend"}
    if stat not in allowed:
        raise ValueError(
            f"plots.individual_plots.time_stat must be one of {sorted(allowed)}. Got: {stat}"
        )
    return stat


def _selected_models(plot_cfg) -> list[str]:
    # gets desired models
    if plot_cfg.models is None:
        return []
    return list(plot_cfg.models)


def _validate_time_order(start: str, end: str):
    # makes sure that the requested time is in the right order (start < end)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError(f"Invalid time range: end ({end}) must not be earlier than start ({start}).")


def _count_requested_steps(start: str, end: str, freq: str) -> int:
    """
    counts how many timesteps are requested between start & end
    -> serves for validating the user's setting in _validate_time_selection_for_method
    (e.g. timeseries needs >= 2 steps, but maps work for 1) 
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if freq == "monthly":
        return (end_ts.year - start_ts.year) * 12 + (end_ts.month - start_ts.month) + 1
    if freq == "daily":
        return (end_ts.normalize() - start_ts.normalize()).days + 1
    raise ValueError(f"Unsupported frequency: {freq}. Expected 'monthly' or 'daily'.")


def _validate_time_selection_for_method(start: str, end: str, freq: str, method: str):
    """
    validates the time selection
    timeseries need to comprise at least 2 timesteps, maps work for a minimum of 1
    """
    n_steps = _count_requested_steps(start, end, freq)

    if method == "timeseries" and n_steps < 2:
        unit = "months" if freq == "monthly" else "days"
        raise ValueError(
            f"For method='timeseries' and freq='{freq}', the selected period must span at least 2 {unit}. "
            f"Got start={start}, end={end}."
        )
    if method == "map" and n_steps < 1:
        unit = "month" if freq == "monthly" else "day"
        raise ValueError(
            f"For method='map' and freq='{freq}', the selected period must span at least 1 {unit}. "
            f"Got start={start}, end={end}."
        )
    

# ---- TIME HELPERS ----
def _format_time_from_freq(ts, freq: str) -> str:
    # formats time correctly, depending on frequence from config
    ts = pd.Timestamp(ts)
    if freq == "monthly":
        return ts.strftime("%Y-%m")
    if freq == "daily":
        return ts.strftime("%Y-%m-%d")
    raise ValueError(f"Unsupported frequence: {freq}. Expected 'monthly' or 'daily'.")


def _selection_bounds_for_freq(start: str, end: str, freq: str) -> tuple[str, str]:
    """
    converts requested dates to selection bounds matching the data frequency
    monthly:
      use full months, e.g. 2024-12-03 .. 2024-12-31 -> 2024-12-01 .. 2024-12-31
    daily:
      normalise to calendar days
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    if freq == "monthly":
        start_sel = start_ts.replace(day=1)
        end_sel = end_ts + pd.offsets.MonthEnd(0)
    elif freq == "daily":
        start_sel = start_ts.normalize()
        end_sel = end_ts.normalize()
    else:
        raise ValueError(f"Unsupported frequency: {freq}. Expected 'monthly' or 'daily'.")

    return str(start_sel.date()), str(end_sel.date())


def _nearest_time_str(da: xr.DataArray, requested: str, freq: str) -> str:
    # gets the nearest timestep
    req = np.datetime64(requested)
    idx = int(np.argmin(np.abs(da.time.values - req)))
    return _format_time_from_freq(da.time.values[idx], freq)


def _time_label(start: str, end: str, method: str, da: xr.DataArray, freq: str, plot_cfg) -> str:
    """
    constructs a time label for the plot titles
    - for "annual_mean" or "trend": show only yyyy
    - for raw data: format timestamps according to frequency: if freq=monthly: yyyy-mm, if freq=daily: yyyy-mm-dd
    - if start == end: use nearest available dataset timestep
    - for maps over multiple timesteps: append "(time mean)"
    """
    stat = _time_stat(plot_cfg)
    if stat in {"annual_mean", "trend"}:
        start_y = pd.Timestamp(start).year
        end_y = pd.Timestamp(end).year
        return f"{start_y} to {end_y}"
    if pd.Timestamp(start) == pd.Timestamp(end):
        return _nearest_time_str(da, start, freq)
    start_str = _format_time_from_freq(start, freq)
    end_str = _format_time_from_freq(end, freq)
    if method == "map":
        return f"{start_str} to {end_str} (time mean)"
    return f"{start_str} to {end_str}"


# ---- GENERIC HELPERS ----
def _as_float_or_none(value) -> float | None:
    # converts a value to float, unless value is None
    if value is None:
        return None
    return float(value)


def _wrap_lon_360(lon: float) -> float:
    # converts values from [-180,180] to 0...360
    return float(lon) % 360.0


def _coord_to_dms_tag(coord: float, axis: str) -> str:
    """
    converts lat/lon config setting into suitable filenames
    e.g.: 50.234923 -> 50-14-06N, -12.5 -> 12-30-00S
    """
    coord = float(coord)
    sign = 1 if coord >= 0 else -1
    coord_abs = abs(coord)
    deg = int(coord_abs)
    minutes_full = (coord_abs - deg) * 60
    minutes = int(minutes_full)
    seconds = int(round((minutes_full - minutes) * 60))
    # fix rounding overflow like 59.9999 -> 60
    if seconds == 60:
        seconds = 0
        minutes += 1
    if minutes == 60:
        minutes = 0
        deg += 1
    if axis == "lat":
        hemisphere = "N" if sign >= 0 else "S"
    elif axis == "lon":
        hemisphere = "E" if sign >= 0 else "W"
    else:
        raise ValueError("axis must be 'lat' or 'lon'")
    return f"{deg}-{minutes:02d}-{seconds:02d}{hemisphere}"


# ---- SPATIAL SUBSETTING HELPERS ----
def _select_bbox(da: xr.DataArray, lat0: float, lat1: float, lon0: float, lon1: float) -> xr.DataArray:
    # relevant for location: individual; extracts data for the requested bounding box
    lat0 = float(lat0)
    lat1 = float(lat1)
    lon0 = _wrap_lon_360(lon0) # ensures longitude is within 0-360
    lon1 = _wrap_lon_360(lon1)

    lat_sel = da.sel(lat=_lat_slice(da, lat0, lat1)) # latitude selection
    if lat_sel.sizes.get("lat", 0) == 0:
        raise ValueError(
            f"Selected latitude range is empty: lat0={lat0}, lat1={lat1}. "
            "Please increase the distance between lat0 and lat1 for a proper map."
        )
    # special case: lon0 ≈ lon1 -> select a single grid column
    # using nearest neighbour to ensure a grid cell is selected
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
    if lon0 < lon1: # normal longitude slice (no dateline crossing)
        return lat_sel.sel(lon=slice(lon0, lon1))
    else: # bounding box crosses dateline (e.g. 350 -> 20°)
        part1 = lat_sel.sel(lon=slice(lon0, 360))
        part2 = lat_sel.sel(lon=slice(0, lon1))
        out = xr.concat([part1, part2], dim="lon") # combine both parts along longitude
        _, unique_idx = np.unique(out["lon"].values, return_index=True) # remove duplicated longitude values
        out = out.isel(lon=np.sort(unique_idx))
    # safety check: ensures selected region indeed contains grid cells
    if out.sizes.get("lon", 0) == 0 or out.sizes.get("lat", 0) == 0:
        raise ValueError(
            f"Selected bounding box contains no grid cells: "
            f"lat0={lat0}, lat1={lat1}, lon0={lon0}, lon1={lon1}."
        )
    return out


def _subset_for_location(da: xr.DataArray, plot_cfg) -> xr.DataArray:
    """
    creates a subset for a given location
    supported modes:
        - None/global -> returns full field
        - "individual" -> selects user-defined lat/lon bounding box
        - "arctic" -> selects latitudes from min_latitude to 90°
        - "antarctic" -> selects latitude from -90° to max_latitude
    """
    location = _normalise_location(plot_cfg.location) # converts config value to accepted location mode

    if location is None: # global: no spatial subsetting
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


def _area_mean(da: xr.DataArray) -> xr.DataArray:
    # computes area-weighted spatial mean over lat & lon
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"Expected 'lat' and 'lon' dimensions, got {da.dims}")
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights).mean(dim=("lat", "lon"))


# ---- DATA TRANSFORMATION HELPERS ----
def _maybe_detrend(da: xr.DataArray, plot_cfg, start: str, end: str) -> xr.DataArray:
    # apply detrending if enabled in the config; otherwise return the input unchanged
    # the trend is always removed along the time dimension
    if not plot_cfg.detrend.enabled:
        return da
    
    if plot_cfg.detrend.base_period == "unique":
        start, end = _selection_bounds_for_freq(plot_cfg.detrend.base_start, plot_cfg.detrend.base_end, plot_cfg.freq)
    elif plot_cfg.detrend.base_period == "total":
        start = start
        end = end
    else:
        raise ValueError(f"Received invalid option for detrending base period {plot_cfg.detrend.base_period}, only accepts 'unique' or 'total'.")
    preserve_mean = bool(plot_cfg.detrend.preserve_mean)
    base_period = plot_cfg.detrend.base_period
    if base_period is not None:
        base_period = tuple(base_period)
    return detrend_dataarray(
        da,
        dim="time",
        start=start,
        end=end,
        preserve_mean=preserve_mean,
    )


def _prepare_field(da: xr.DataArray, plot_cfg, method: str, start: str, end: str) -> xr.DataArray:
    """
    prepares the data array for plotting depending on the configured time statistic (raw, annual_mean, trend), detrending, and plotting method (map, timeseries)
    rules for detrending:
        - raw:
            * timeseries: detrend after area mean
            * map: detrend each grid point before time averaging
        - annual mean:
            * timeseries: area mean -> annual mean -> detrend
            * map: anuual mean at each grid point -> detrend annual series per grid point -> mean over time
    """
    stat = _time_stat(plot_cfg) # get "raw, annual_mean, or trend"
    da_loc = _subset_for_location(da, plot_cfg) # create subset if requested
    if plot_cfg.anomaly:
        da_loc, _ = to_anomaly(da_loc, plot_cfg.baseline.start, plot_cfg.baseline.end) # convert to anomaly relative to configured baseline

    if stat == "raw":
        if method == "map":
            # map uses the temporal mean of the selected period
            if da_loc.sizes.get("time", 0) == 0:
                raise ValueError("No timesteps remain after time selection.")
            da_plot = _maybe_detrend(da_loc, plot_cfg, start, end)
            return da_plot.mean(dim="time")
        if method == "timeseries":
            # timeseries uses the spatial mean
            da_series = _area_mean(da_loc)
            da_series = _maybe_detrend(da_series, plot_cfg, start, end)
            return da_series

    elif stat == "annual_mean":
        if method == "map":
            # compute annual means first, then average those years spatially
            da_ann = annual_weighted_mean(da_loc)
            if da_ann.sizes.get("time", 0) == 0:
                raise ValueError("No annual timesteps remain after aggregation.")
            da_ann = _maybe_detrend(da_ann, plot_cfg, start, end)
            return da_ann.mean(dim="time")

        if method == "timeseries":
            # compute spatial mean first, then annual mean timeseries
            da_series = annual_weighted_mean(_area_mean(da_loc))
            da_series = _maybe_detrend(da_series, plot_cfg, start, end)
            return da_series

    elif stat == "trend":
        if method == "map":
            # compute per-gridpoint linear trend (converted to decadal trend)
            if da_loc.sizes.get("time", 0) < 2:
                raise ValueError("Need at least 2 timesteps to compute a trend map.")
            return compute_slope_per_gridpoint(da_loc) * 10.0

        if method == "timeseries":
            # same as for annual_mean + timeseries; the decadal trend line is computed and added later during plotting
            return annual_weighted_mean(_area_mean(da_loc))

    raise ValueError(f"Unsupported method/stat combination: method={method}, time_stat={stat}")


def _standardise_time_for_difference(da: xr.DataArray, plot_cfg) -> xr.DataArray:
    """
    standardises the time coordinate before computing model - ERA5 difference
    ensures that both datasets use comparable time indices 
    depends on stat and freq from user 
    """
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
    # calculates the difference between the model data and ERA5
    if "time" in model_da.dims and "time" in era5_da.dims:
        model_da = _standardise_time_for_difference(model_da, plot_cfg)
        era5_da = _standardise_time_for_difference(era5_da, plot_cfg)

        model_da, era5_da = xr.align(model_da, era5_da, join="inner")

        if model_da.sizes.get("time", 0) == 0:
            raise ValueError(
                "No overlapping timesteps remain after aligning model and ERA5 for difference plot. "
                f"time_stat={_time_stat(plot_cfg)}, freq={plot_cfg.freq}"
            )

    return model_da - era5_da



# ---- PLOT RANGE / COLOURBAR HELPERS ----
def _summary_column_prefix(single_time: bool) -> str:
    # just renaming to match columns in csv
    return "raw" if single_time else "temporal"


def _dynamic_bounds(arrays: list[xr.DataArray], percentile=99, symmetric=False) -> tuple[float, float]:
    """
    computes the colourbar bounds dynamically from the data when NO csv-based bounds are available
    collects all relevant values and determines vmin/vmax based on a percentile rule 
        - 99 = 1st-99th percentile
        - 95 = 5th-95th percentile
        - raw = full min/max range
    """
    vals = []
    # gets all values needed
    for da in arrays:
        data = np.asarray(da.values).ravel()
        data = data[np.isfinite(data)]
        if data.size:
            vals.append(data)
    if not vals:
        raise ValueError("Could not determine plotting range because all arrays are empty or NaN.")
    vals = np.concatenate(vals)
    # determines bounds according to selected percentile
    perc = str(percentile).lower()
    if perc == "99":
        vmin, vmax = np.nanpercentile(vals, [1, 99])
    elif perc == "95":
        vmin, vmax = np.nanpercentile(vals, [5, 95])
    elif perc == "raw":
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    else:
        raise ValueError(f"Unknown percentile option: {percentile}")

    if symmetric: # if symmetric=True, the bounds are forced symmetrically around zero
        vmax_abs = max(abs(float(vmin)), abs(float(vmax)))
        if vmax_abs == 0:
            vmax_abs = 1e-12
        return -vmax_abs, vmax_abs

    if float(vmin) == float(vmax): # prevent identical bounds
        eps = max(abs(float(vmin)) * 0.05, 1e-12)
        return float(vmin) - eps, float(vmax) + eps
    return float(vmin), float(vmax)


def _get_map_bounds(cfg, plot_cfg, arrays: list[xr.DataArray], var: str, plev, difference: bool, anomaly: bool, single_time: bool) -> tuple[float, float]:
    """
    determines the map colourbar bounds, preferably from precomputed csv file
    uses 'csv_file1' for absolute model/ERA5 plots, and 'csv_file2' for difference plots
    chooses the correct csv column prefix depeding on plot type:
        - trend maps -> slope
        - single-time map -> raw
        - multiple-time map -> temporal
    if csv lookup fails or anomaly = True, falls back to _dynamic_bounds
    """
    use_csv = not anomaly # does not work for anomaly as user can input so many different baselines that no bounds are precomputed for these cases
    if use_csv:
        try:
            csv_rel = plot_cfg.range_source.csv_file2 if difference else plot_cfg.range_source.csv_file1
            csv_path = os.path.join(hydra.utils.get_original_cwd(), csv_rel)
            stat = _time_stat(plot_cfg)
            if stat == "trend":
                prefix = "slope"
            else:
                prefix = _summary_column_prefix(single_time)
            vmin, vmax = get_range_from_csv(
                percentile=plot_cfg.range_source.percentile,
                csv_file=csv_path,
                var=var,
                plev=plev,
                prefix=prefix,
            )
            if difference:
                # vmax_abs = max(abs(vmin), abs(vmax))
                # return -vmax_abs, vmax_abs
                return vmin, vmax
            return vmin, vmax
        except Exception as exc:
            print(f"Falling back to dynamic colour range for {var}, plev={plev}: {exc}")

    return _dynamic_bounds(
        arrays,
        percentile=plot_cfg.range_source.percentile,
        symmetric=difference or anomaly,
    )


def _use_zero_centered_bins(plot_cfg, vmin: float, vmax: float) -> bool:
    # gets the colourbar mode: centered, linear or auto
    mode = str(getattr(plot_cfg.colourbar, "mode", "auto")).strip().lower()
    if mode not in {"auto", "centered", "linear"}:
        raise ValueError(
            f"Unknown colourbar mode: {mode}. Use one of: auto, centered, linear"
        )
    if mode == "linear":
        return False
    if mode == "centered":
        if not (vmin <= 0 <= vmax):
            raise ValueError(
                f"colourbar.mode='centered' requires 0 to lie within the plotting range, "
                f"but got vmin={vmin}, vmax={vmax}. "
                "Use mode='linear' for absolute fields like geopotential height."
            )
        return True
    # auto
    return bool(plot_cfg.difference or plot_cfg.anomaly or (vmin <= 0 <= vmax))


def _map_levels_and_ticks(vmin: float, vmax: float, plot_cfg) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    determines discrete colourbar bins (levels) and ticks for map plots when use_custom_bins are enabled
    if disabled, returns (None, None)
    """
    cbar_cfg = plot_cfg.colourbar

    if not cbar_cfg.use_custom_bins:
        return None, None
    # determine bin size either explicitely of from requested number of bin
    if cbar_cfg.bin_size is None:
        bin_size = nice_bin_size(vmin, vmax, cbar_cfg.target_bins)
    else:
        bin_size = float(cbar_cfg.bin_size)

    zero_centered = _use_zero_centered_bins(plot_cfg, vmin, vmax)

    if zero_centered:
        # constructs symmetric levels with a white bin around zero
        levels = build_zero_bin_levels(vmin=vmin, vmax=vmax, bin_size=bin_size) 
        # computes symmetric colourbar ticks based on the levels
        ticks = symmetric_ticks_from_levels(
            levels,
            vmin,
            vmax,
            keep_every=int(cbar_cfg.tick_every),
            include_zero=bool(cbar_cfg.include_zero_tick),
        )
    else:
        levels = np.arange(vmin, vmax + bin_size, bin_size)
        if levels[-1] < vmax:
            levels = np.append(levels, vmax)
        tick_every = int(cbar_cfg.tick_every)
        ticks = levels[::tick_every]
        if ticks[-1] != levels[-1]:
            ticks = np.append(ticks, levels[-1])

    return levels, ticks


def _get_map_norm(plot_cfg, vmin, vmax):
    # gets the correct norm for the colourbar, depending if bins shall be centered around zero
    zero_centered = _use_zero_centered_bins(plot_cfg, vmin, vmax)
    if zero_centered:
        return mpl.colors.CenteredNorm(vcenter=0)
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax)


# ---- PLOT LAYOUT HELPERS ----
def _projection_and_extent(plot_cfg):
    # chooses the map projection and geographic extent based on the location mode in the config
    location = _normalise_location(plot_cfg.location)
    centre = float(getattr(plot_cfg.global_centre, 0))
    if location is None: # global = Robinson projection
        return ccrs.Robinson(central_longitude=centre), None
    if location == "individual":
        lon0 = _wrap_lon_360(float(plot_cfg.individual.lon0))
        lon1 = _wrap_lon_360(float(plot_cfg.individual.lon1))
        # special case: treat nearly identical longitudes as point selection & add small padding so the map remains visible
        if math.isclose(lon0, lon1):
            pad = float(getattr(plot_cfg.individual, "point_pad_deg", 2.0))
            extent = [lon0 - pad, lon0 + pad, float(plot_cfg.individual.lat0) - pad, float(plot_cfg.individual.lat1) + pad]
        else: # regular bounding box; if lon0>lon1 extrend across dateline
            lon_min = lon0
            lon_max = lon1
            if lon0 > lon1:
                lon_max += 360.0
            extent = [lon_min, lon_max, min(float(plot_cfg.individual.lat0), float(plot_cfg.individual.lat1)), max(float(plot_cfg.individual.lat0), float(plot_cfg.individual.lat1))]
        return ccrs.PlateCarree(), extent # plateCarree for standard map of just a section
    if location == "arctic":
        return ccrs.NorthPolarStereo(), [-180, 180, float(plot_cfg.polar.min_latitude), 90] # arctic projection
    if location == "antarctic":
        return ccrs.SouthPolarStereo(), [-180, 180, -90, float(plot_cfg.polar.max_latitude)] # antarctic projection
    return ccrs.Robinson(central_longitude=centre), None


def _resolve_figsize(plot_cfg, method: str):
    """
    determines the figure size
    order:
    1. if figsize is set in config: use it
    2. if map + polar: [8, 8]
    3. if map + global: [9, 5]
    4. if timeseries: [12, 4]
    5. otherwise: let matplotlib decide (None)
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


# ---- PLOT RENDERING HELPERS ----
def _plot_single_map(ax, da: xr.DataArray, title: str, cfg, plot_cfg, vmin: float, vmax: float, levels=None):
    """
    draws a single map panel on the given axis
    applies coastlines, gridlines, configured extent and filled contours 
    """
    projection, extent = _projection_and_extent(plot_cfg)
    location = _normalise_location(plot_cfg.location)

    ax.set_title(title, fontsize=10)
    ax.coastlines(linewidth=0.9, color=str(plot_cfg.coastline_colour))

    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree()) # restrict plot to configured extent if needed

    ax.gridlines(draw_labels=True, linewidth=0.5, color="black", alpha=0.35, linestyle="--")

    # for global and polar maps only: add cyclic point to avoid seam at 0/360°
    if location is None or location in POLAR_LOCATIONS:
        data_cyc, lon_cyc = add_cyclic_point(da.values, coord=da["lon"].values)
        cf = ax.contourf(
            lon_cyc,
            da["lat"].values,
            data_cyc,
            norm=_get_map_norm(plot_cfg, vmin, vmax), #mpl.colors.CenteredNorm(vcenter=0),
            levels=levels, #np.linspace(vmin, vmax, 21),
            cmap=str(plot_cfg.colour_scheme),
            extend="both",
            transform=ccrs.PlateCarree(),
        )
    else:
        # regional maps need at least a small 2D grid to be plottable
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
            norm=_get_map_norm(plot_cfg, vmin, vmax), #mpl.colors.CenteredNorm(vcenter=0),
            levels=levels, #np.linspace(vmin, vmax, 21),
            cmap=str(plot_cfg.colour_scheme),
            extend="both",
            transform=ccrs.PlateCarree(),
        )
    return cf


def _plot_timeseries(ax, era5_series: xr.DataArray, model_series: xr.DataArray, plot_cfg,
                    model_name: str, proper_model_name: str, member: str,  unit_here: str):
    """
    draws a single timeseries panel for ERA5 and one model member or mean
    if time_stat is "trend", linear regression is added
    if difference=True, only model - ERA5 is shown and a zero reference line is added
    """
    colours = plot_cfg.colours
    base_colour = colours.base_colours[model_name]
    light_colour = colours.colours_light[model_name]
    stat = _time_stat(plot_cfg)
    if stat == "trend":
        # for trend plots; shows annual-mean series AND corresponding regression lines
        # ERA5 is only shown if not plotting difference
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
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8) # add horizontal line at 0 for reference

    else:
        # for raw/annual_nmean plots: either show difference series only, or ERA5 and model together
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
    if plot_cfg.ylabel is not None:
        ax.set_ylabel(str(plot_cfg.ylabel))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc=str(plot_cfg.legend_loc), frameon=False)


# ---- TITLE / FILENAME HELPERS ----
def _default_title(plot_cfg, method: str, long_name: str, proper_model_name: str, member: str, time_label: str, plev_title: str):
    # builds the default plot title when no custom title template is provided in the config
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
    # add baseline information as second line if anomaly is used
    if plot_cfg.anomaly:
        base_start = _format_time_from_freq(plot_cfg.baseline.start, plot_cfg.freq)
        base_end = _format_time_from_freq(plot_cfg.baseline.end, plot_cfg.freq)
        title += f"\nBaseline removed: mean of {base_start} – {base_end}"
    # add detrending information as 2nd/3rd line if anomaly is used
    if plot_cfg.detrend.enabled and plot_cfg.detrend.base_period == "unique":
        base_start = _format_time_from_freq(plot_cfg.detrend.base_start, plot_cfg.freq)
        base_end = _format_time_from_freq(plot_cfg.detrend.base_end, plot_cfg.freq)
        mean = ", mean of entire time period readded" if plot_cfg.detrend.preserve_mean else ""
        title += f"\nDetrended over {base_start} – {base_end}{mean}"
    elif plot_cfg.detrend.enabled:
        mean = ", mean readded" if plot_cfg.detrend.preserve_mean else ""
        title += f"\nDetrended over entire time period{mean}"

    return title


def _format_title(plot_cfg, method: str, var: str, long_name: str, model_name: str, proper_model_name: str, member: str, plev, plev_title: str, start: str, end: str, time_label: str):
    # returns the final plot title;
    # if a custom title template is provided in the config, fill it dynamically, otherwise use internally generated default title
    if plot_cfg.title:
        # convert pressure level to hPa
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
    # otherwise construct default title
    return _default_title(plot_cfg, method, long_name, proper_model_name, member, time_label, plev_title)


def _output_filename(method: str, var: str, plev_tag: str, model_name: str, member: str, 
                     start: str, end: str, plot_cfg) -> str:
    # creates detailed filenames that contain all relevant information
    if model_name == "era5":
        model_tag = "era5"
    else:
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
    if stat == "trend":
        stat_tag = "_decadalTrend" 
    elif plot_cfg.detrend.enabled:
        stat_tag = "_detrended" 
    else: 
        stat_tag = ""
    member = f"_{member}"
    return f"{method}_{var}{plev_tag}_{model_tag}{member}_{loc_tag}{diff_tag}{anom_tag}_{start_tag}-{end_tag}{stat_tag}.png"


# ---- ENSEMBLE / MEMBER HANDLING HELPER ----
def _prepare_member_mapping(cfg, plot_cfg, member_to_da: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    # builds the dictionary of members to plot
    # optionally compute and add ensemble mean as an extra entry
    if plot_cfg.include_ensemble_mean_as_member:
        member_to_da = ensemble_mean_as_member(member_to_da, name="mean")

    if plot_cfg.only_mean: # if only ensemble mean should be plotted, only return that entry
        if not plot_cfg.include_ensemble_mean_as_member:
            raise ValueError(
                "plots.individual_plots.only_mean=true requires include_ensemble_mean_as_member=true."
            )
        if "mean" not in member_to_da:
            raise ValueError("Requested only_mean=true, but ensemble mean could not be constructed.")
        return {"mean": member_to_da["mean"]}
    # otherwise return all configured members
    out = {m: member_to_da[m] for m in cfg.members}
    if plot_cfg.include_ensemble_mean_as_member:
        out["mean"] = member_to_da["mean"] # append mean as additional plotting entry if requested
    return out


# ---- MAIN ----
def run(cfg):
    """
    entry point for individual_plots
    workflow:
        1. validate global plot settings
        2. loop over variables and pressure levels
        3. load/prepare ERA5 reference data
        4. load/prepare model members and generate model plots
        5. optionally generate ERA5-only maps 
    """
    # 1. read config and validate global settings
    plot_cfg = cfg.plots.individual_plots
    method = str(plot_cfg.method).strip().lower() if plot_cfg.method is not None else None
    if method not in {"timeseries", "map"}:
        raise ValueError("plots.individual_plots.method must be either 'timeseries' or 'map'.")
    if method == "map" and plot_cfg.map_era5 and plot_cfg.difference:
        print(
            "Skipping ERA5 map in individual_plots because map_era5=true and difference=true (ERA5 - ERA5 will be zero)."
        )
    if _time_stat(plot_cfg) == "trend" and plot_cfg.detrend.enabled:
        raise ValueError(
            "Detrending cannot be used together with time_stat='trend', because that would remove the trend you want to analyse."
        )
    figsize = _resolve_figsize(plot_cfg, method)

    # 2. iterate over variables & pressure levels
    for item in iter_vars_and_plevs(cfg, plot_cfg):
        var = item["var"]
        long_name = item["long_name"]
        unit = item["unit"]
        start = item["start"]
        end = item["end"]
        _validate_time_order(start, end)
        _validate_time_selection_for_method(start, end, plot_cfg.freq, method)
        start_sel, end_sel = _selection_bounds_for_freq(start, end, plot_cfg.freq)
        ensure_allowed_var(cfg, var)

        for plev in item["plevs"]:
            plev_title, plev_tag = plev_strings(plev)
            # 3. load and prepare era5 data
            era5 = open_era5_da(cfg, var=var, start=start_sel, end=end_sel, plev=plev)
            era5, unit_here = conversion_rules(var, era5, cfg, "era5", unit)
            era5_prepared = _prepare_field(era5, plot_cfg, method, start_sel, end_sel)
            time_label = _time_label(start, end, method, era5, plot_cfg.freq, plot_cfg)
            single_time = pd.Timestamp(start) == pd.Timestamp(end)

            # 4. model plotting part
            for model_name in _selected_models(plot_cfg):
                model_cfg = cfg.datasets.models[model_name]
                proper_model_name = getattr(model_cfg, "proper_name", model_name)
                # 4a. open and prepare members of models
                member_to_da = {}
                for member in cfg.members:
                    da_model = open_model_da(
                        model_cfg=model_cfg,
                        cfg=cfg,
                        member=member,
                        var=var,
                        modelname=model_cfg.modelname,
                        freq=plot_cfg.freq,
                        start=start_sel,
                        end=end_sel,
                        grid=plot_cfg.grid,
                        plev=plev,
                    )
                    da_model, _ = conversion_rules(var, da_model, cfg, "model", unit_here)
                    prepared = _prepare_field(da_model, plot_cfg, method, start_sel, end_sel)
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
                # 4b. TIMESERIES plotting
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
                # 4c. MAP plotting
                else:
                    arrays = list(member_to_plot.values())
                    if plot_cfg.colourbar.manual_vmin and plot_cfg.colourbar.manual_vmax:
                        vmin = float(plot_cfg.colourbar.manual_vmin)
                        vmax = float(plot_cfg.colourbar.manual_vmax)
                    else:
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
                    print(f"diff: {bool(plot_cfg.difference)}, anomaly: {bool(plot_cfg.anomaly)}, vmin: {vmin}, vmax: {vmax}")
                    if plot_cfg.colourbar.use_custom_bins:
                        levels, ticks = _map_levels_and_ticks(vmin, vmax, plot_cfg)
                    else:
                        levels = np.linspace(vmin, vmax, 21)
                        ticks = None
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
                        cf = _plot_single_map(ax, map_da, "", cfg, plot_cfg, vmin=vmin, vmax=vmax, levels=levels)
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
                        if ticks is not None:
                            cbar.set_ticks(ticks)
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
            # 5. optional ERA5 map
            if method == "map" and plot_cfg.map_era5 and not plot_cfg.difference:
                outdir = os.path.join(
                    hydra.utils.get_original_cwd(),
                    cfg.out.dir,
                    "individual_plots",
                    method,
                    var,
                )
                os.makedirs(outdir, exist_ok=True)
                if plot_cfg.colourbar.manual_vmin and plot_cfg.colourbar.manual_vmax:
                    vmin = float(plot_cfg.colourbar.manual_vmin)
                    vmax = float(plot_cfg.colourbar.manual_vmax)
                else:
                    vmin, vmax = _get_map_bounds(
                        cfg,
                        plot_cfg,
                        [era5_prepared],
                        var,
                        plev,
                        difference=False,
                        anomaly=bool(plot_cfg.anomaly),
                        single_time=single_time,
                    )

                if plot_cfg.colourbar.use_custom_bins:
                    levels, ticks = _map_levels_and_ticks(vmin, vmax, plot_cfg)
                else:
                    levels = np.linspace(vmin, vmax, 21)
                    ticks = None

                projection, _ = _projection_and_extent(plot_cfg)

                fname = _output_filename(method, var, plev_tag, "era5", "", start, end, plot_cfg)
                outfile = os.path.join(outdir, fname)

                if not (cfg.out.savefig and not should_compute_output(outfile, getattr(cfg.out, "overwrite", "ask"))):
                    fig, ax = plt.subplots(
                        figsize=figsize,
                        subplot_kw={"projection": projection},
                    )
                    fig.subplots_adjust(top=0.9, bottom=0.12)

                    cf = _plot_single_map(ax, era5_prepared, "", cfg, plot_cfg, vmin=vmin, vmax=vmax, levels=levels)

                    title = _format_title(
                        plot_cfg,
                        method,
                        var,
                        long_name,
                        "era5",
                        "ERA5",
                        "",
                        plev,
                        plev_title,
                        start,
                        end,
                        time_label,
                    )
                    ax.set_title(title, pad=13)

                    cbar = fig.colorbar(cf, ax=ax, orientation="horizontal", pad=0.1, shrink=0.9)
                    if ticks is not None:
                        cbar.set_ticks(ticks)

                    if _time_stat(plot_cfg) == "trend":
                        cbar_label = f"{long_name} ({unit_here}/decade)"
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
