from __future__ import annotations

import os
import warnings
from typing import Iterable

import cartopy.crs as ccrs
import hydra
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.util import add_cyclic_point

from evaluation.general_functions import (
    ensemble_mean_as_member,
    model_abbrev,
    normalise_plevs,
    open_era5_da,
    open_model_da,
    plev_strings,
    resolve_period,
    should_compute_output,
)
from evaluation.metrics.individual_plots import (
    _normalise_location,
    _projection_and_extent,
    _resolve_figsize,
    _subset_for_location,
)

POLAR_LOCATIONS = {"arctic", "antarctic"}
CYCLIC_POINT_NEEDED = {None, "ortho", "arctic", "antarctic"}
ALLOWED_PLEVS_PA = [5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000]
ALLOWED_PLEVS_HPA = [int(v / 100) for v in ALLOWED_PLEVS_PA]
SEASON_MONTHS = {
    "full": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

def _normalise_plev_entries(plev_cfg) -> list:
    raw = normalise_plevs(plev_cfg)
    out = []
    for item in raw:
        if isinstance(item, str) and item.strip().lower() == "surface":
            out.append("surface")
        else:
            out.append(item)
    return out


def _plev_entries(plot_cfg) -> list:
    return _normalise_plev_entries(plot_cfg.plev)


def _background_mode(plot_cfg) -> str:
    mode = str(getattr(plot_cfg, "background", "speed")).strip().lower()
    if mode not in {"speed", "pressure"}:
        raise ValueError(
            f"Unsupported wind.background option: {mode!r}. Allowed options are 'speed' and 'pressure'."
        )
    return mode


def _background_long_name(plot_cfg) -> str:
    bg_mode = _background_mode(plot_cfg)
    if bg_mode == "speed":
        return "Wind speed"
    if bg_mode == "pressure":
        return "Sea-level pressure"
    raise ValueError(f"Unsupported background mode: {bg_mode}")


def _background_cbar_label(plot_cfg, plev_label: str) -> str:
    bg_mode = _background_mode(plot_cfg)

    if bg_mode == "speed":
        if plot_cfg.difference:
            return f"Wind speed difference at {plev_label} (m s$^{{-1}}$)"
        return f"Mean wind speed at {plev_label} (m s$^{{-1}}$)"

    if bg_mode == "pressure":
        if plot_cfg.difference:
            return "Sea-level pressure difference (Pa)"
        return "Mean sea-level pressure (Pa)"

    raise ValueError(f"Unsupported background mode: {bg_mode}")


def _wind_component_names(plev):
    if plev == "surface":
        return "uas", "vas"
    return "ua", "va"


def _selected_season(plot_cfg) -> str:
    season = str(getattr(plot_cfg, "season", "full")).strip()
    if season not in SEASON_MONTHS:
        raise ValueError(
            f"plots.wind.season must be one of {list(SEASON_MONTHS.keys())}. Got: {season}"
        )
    return season


def _subset_time_for_season(da: xr.DataArray, plot_cfg) -> xr.DataArray:
    season = _selected_season(plot_cfg)
    if season == "full":
        return da
    if "time" not in da.dims:
        raise ValueError(f"Expected a 'time' dimension for seasonal subsetting, got dims={da.dims}")
    months = SEASON_MONTHS[season]
    out = da.where(da["time"].dt.month.isin(months), drop=True)
    if out.sizes.get("time", 0) == 0:
        raise ValueError(
            f"No timesteps left after applying season={season!r}. "
            "Check the selected start/end period and data coverage."
        )
    return out


def _season_tag(plot_cfg) -> str:
    season = _selected_season(plot_cfg)
    return "" if season == "full" else f"_{season}"


def _season_label(plot_cfg) -> str:
    season = _selected_season(plot_cfg)
    return "" if season == "full" else f" | {season}"


def _member_mean_mapping(plot_cfg, cfg, member_to_da: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    if plot_cfg.include_ensemble_mean_as_member:
        member_to_da = ensemble_mean_as_member(member_to_da, name="mean")

    if plot_cfg.only_mean:
        if not plot_cfg.include_ensemble_mean_as_member:
            raise ValueError("wind.only_mean=true requires wind.include_ensemble_mean_as_member=true.")
        if "mean" not in member_to_da:
            raise ValueError("Requested only_mean=true, but ensemble mean could not be constructed.")
        return {"mean": member_to_da["mean"]}

    out = {m: member_to_da[m] for m in cfg.members}
    if plot_cfg.include_ensemble_mean_as_member:
        out["mean"] = member_to_da["mean"]
    return out


def _time_average(da: xr.DataArray, plot_cfg) -> xr.DataArray:
    if "time" not in da.dims:
        raise ValueError(f"Expected a 'time' dimension, got dims={da.dims}")
    da = _subset_time_for_season(da, plot_cfg)
    return da.mean(dim="time")


def _compute_scalar_and_vector_climatology(u: xr.DataArray, v: xr.DataArray, plot_cfg) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    speed = np.sqrt(u**2 + v**2)
    speed_clim = _time_average(speed, plot_cfg)
    u_clim = _time_average(u, plot_cfg)
    v_clim = _time_average(v, plot_cfg)
    return speed_clim, u_clim, v_clim


def _prepare_model_member_fields(model_cfg, cfg, plot_cfg, member: str, plev, start: str, end: str):
    u_var, v_var = _wind_component_names(plev)
    plev_arg = None if plev == "surface" else plev

    u = open_model_da(
        model_cfg=model_cfg,
        cfg=cfg,
        member=member,
        var=u_var,
        modelname=model_cfg.modelname,
        freq=plot_cfg.freq,
        start=start,
        end=end,
        grid=plot_cfg.grid,
        plev=plev_arg,
    )
    v = open_model_da(
        model_cfg=model_cfg,
        cfg=cfg,
        member=member,
        var=v_var,
        modelname=model_cfg.modelname,
        freq=plot_cfg.freq,
        start=start,
        end=end,
        grid=plot_cfg.grid,
        plev=plev_arg,
    )

    return _compute_scalar_and_vector_climatology(u, v, plot_cfg)


def _prepare_era5_fields(cfg, plot_cfg, plev, start: str, end: str):
    u_var, v_var = _wind_component_names(plev)
    plev_arg = None if plev == "surface" else plev

    u = open_era5_da(cfg, var=u_var, start=start, end=end, plev=plev_arg, freq=plot_cfg.freq, grid=plot_cfg.grid)
    v = open_era5_da(cfg, var=v_var, start=start, end=end, plev=plev_arg, freq=plot_cfg.freq, grid=plot_cfg.grid)
    return _compute_scalar_and_vector_climatology(u, v, plot_cfg)


def _prepare_model_background_field(model_cfg, cfg, plot_cfg, member: str, start: str, end: str) -> xr.DataArray:
    bg_mode = _background_mode(plot_cfg)

    if bg_mode == "speed":
        raise ValueError("_prepare_model_background_field should not be called for background='speed'.")

    if bg_mode == "pressure":
        da = open_model_da(
            model_cfg=model_cfg,
            cfg=cfg,
            member=member,
            var="psl",
            modelname=model_cfg.modelname,
            freq=plot_cfg.freq,
            start=start,
            end=end,
            grid=plot_cfg.grid,
            plev=None,
        )
        return _time_average(da, plot_cfg)

    raise ValueError(f"Unsupported background mode: {bg_mode}")


def _prepare_era5_background_field(cfg, plot_cfg, start: str, end: str) -> xr.DataArray:
    bg_mode = _background_mode(plot_cfg)

    if bg_mode == "speed":
        raise ValueError("_prepare_era5_background_field should not be called for background='speed'.")

    if bg_mode == "pressure":
        da = open_era5_da(cfg, var="psl", start=start, end=end, plev=None, freq=plot_cfg.freq, grid=plot_cfg.grid)
        return _time_average(da, plot_cfg)

    raise ValueError(f"Unsupported background mode: {bg_mode}")


def _location_tag(plot_cfg) -> str:
    loc = _normalise_location(plot_cfg.location)
    return "global" if loc is None else str(loc)


def _time_tag(plot_cfg, start: str, end: str) -> tuple[str, str]:
    if str(plot_cfg.freq).strip().lower() == "monthly":
        return start[:7].replace("-", ""), end[:7].replace("-", "")
    return start.replace("-", ""), end.replace("-", "")


def _wind_filename(model_name: str, member: str, plev, plot_cfg, start: str, end: str, era5: bool = False) -> str:
    start_tag, end_tag = _time_tag(plot_cfg, start, end)
    loc_tag = _location_tag(plot_cfg)
    plev_tag = "_surface" if plev == "surface" else plev_strings(plev)[1]
    diff_tag = "_minusERA5" if plot_cfg.difference and not era5 else ""
    model_tag = "era5" if era5 else model_abbrev(model_name)
    member_tag = "" if era5 else f"_{member}"
    bg_tag = _background_mode(plot_cfg)
    season_tag = _season_tag(plot_cfg)
    return f"wind{plev_tag}_{bg_tag}_{model_tag}{member_tag}_{loc_tag}{season_tag}{diff_tag}_{start_tag}-{end_tag}.png"


def _format_title(plot_cfg, model_name: str, proper_model_name: str, member: str, plev, start: str, 
                  end: str, era5: bool = False) -> str:
    bg_mode = _background_mode(plot_cfg)
    bg_long_name = _background_long_name(plot_cfg)

    if plev == "surface":
        plev_label = "surface"
        plev_hpa = None
    else:
        plev_title_str, _ = plev_strings(plev)
        plev_label = plev_title_str.strip().removeprefix("at ").strip()
        plev_hpa = int(float(plev)) if float(plev) < 2000 else int(float(plev) / 100)

    time_label = f"{start} to {end}"
    season_suffix = _season_label(plot_cfg)

    if plot_cfg.title:
        return str(plot_cfg.title).format(
            var="wind",
            long_name=bg_long_name,
            background=bg_mode,
            season=_selected_season(plot_cfg),
            model=("era5" if era5 else model_name),
            proper_model_name=("ERA5" if era5 else proper_model_name),
            member=("" if era5 else member),
            plev=plev,
            plev_hpa=plev_hpa,
            method="map",
            start=start,
            end=end,
            time_label=time_label,
        )

    if bg_mode == "speed":
        if era5:
            return f"ERA5 wind speed{' at ' + plev_label if plev != 'surface' else ' at surface'}{season_suffix} | {time_label}"
        if plot_cfg.difference:
            return f"Wind speed difference to ERA5{' at ' + plev_label if plev != 'surface' else ' at surface'}{season_suffix} | {proper_model_name} {member} | {time_label}"
        return f"Wind speed{' at ' + plev_label if plev != 'surface' else ' at surface'}{season_suffix} | {proper_model_name} {member} | {time_label}"

    if bg_mode == "pressure":
        if era5:
            return f"ERA5 sea-level pressure with wind vectors{' at ' + plev_label if plev != 'surface' else ' at surface'}{season_suffix} | {time_label}"
        if plot_cfg.difference:
            return f"Sea-level pressure difference to ERA5 with wind vectors{' at ' + plev_label if plev != 'surface' else ' at surface'}{season_suffix} | {proper_model_name} {member} | {time_label}"
        return f"Sea-level pressure with wind vectors{' at ' + plev_label if plev != 'surface' else ' at surface'}{season_suffix} | {proper_model_name} {member} | {time_label}"

    raise ValueError(f"Unsupported background mode: {bg_mode}")


def _maybe_manual_bounds(plot_cfg):
    cbar_cfg = plot_cfg.colourbar
    if bool(cbar_cfg.manual):
        if cbar_cfg.manual_vmin is None or cbar_cfg.manual_vmax is None:
            raise ValueError("wind.colourbar.manual=true requires both manual_vmin and manual_vmax.")
        return float(cbar_cfg.manual_vmin), float(cbar_cfg.manual_vmax)
    return None


def _csv_path_from_plot_cfg(cfg, plot_cfg, difference: bool, background: str) -> str:
    if background == "speed":
        csv_rel = plot_cfg.range_source.csv_file2 if difference else plot_cfg.range_source.csv_file1
    elif background == "pressure":
        csv_rel = plot_cfg.range_source.csv_regular2 if difference else plot_cfg.range_source.csv_regular1
    else:
        raise ValueError(f"Unsupported background mode for CSV selection: {background}")

    return os.path.join(hydra.utils.get_original_cwd(), str(csv_rel))


def _csv_percentile_columns(percentile, difference: bool) -> tuple[str | None, str]:
    perc = str(percentile).lower()

    if difference:
        if perc == "raw":
            return "raw_min", "raw_max"
        if perc == "99":
            return "p01", "p99"
        if perc == "95":
            return "p05", "p95"
    else:
        if perc == "raw":
            return None, "raw_max"
        if perc == "99":
            return None, "p99"
        if perc == "95":
            return None, "p95"

    raise ValueError(f"Unknown wind.range_source.percentile option: {percentile}")


def _select_wind_speed_row(df: pd.DataFrame, plev):
    sub = df[df["var"] == "wind_speed"].copy()

    if plev == "surface":
        sub = sub[sub["plev"].astype(str).str.lower() == "surface"]
    else:
        plev_pa = _plev_to_pa_for_csv(plev)
        if "plev_pa" not in sub.columns:
            raise ValueError("Wind-speed bounds CSV is missing required column 'plev_pa'.")
        sub = sub[sub["plev_pa"].notna()]
        sub = sub[sub["plev_pa"].astype(float) == float(plev_pa)]

    if len(sub) == 0:
        raise ValueError(f"No wind_speed row found in CSV for plev={plev!r}.")
    if len(sub) > 1:
        raise ValueError(f"Multiple wind_speed rows found in CSV for plev={plev!r}.")

    return sub.iloc[0]


def _select_regular_row(df: pd.DataFrame, var_name: str, plev):
    sub = df[df["var"].astype(str).str.lower() == str(var_name).lower()].copy()

    if var_name == "psl":
        if "plev" in sub.columns:
            plev_as_str = sub["plev"].astype(str).str.lower()
            surface_like = plev_as_str.isin(["surface", "nan", "none"])
            if surface_like.any():
                sub = sub[surface_like]
        elif "plev_pa" in sub.columns:
            sub = sub[sub["plev_pa"].isna()]
    elif plev == "surface":
        if "plev" in sub.columns:
            plev_as_str = sub["plev"].astype(str).str.lower()
            surface_like = plev_as_str.isin(["surface", "nan", "none"])
            if surface_like.any():
                sub = sub[surface_like]
    else:
        plev_pa = _plev_to_pa_for_csv(plev)

        if "plev_pa" in sub.columns:
            sub = sub[sub["plev_pa"].notna()]
            sub = sub[sub["plev_pa"].astype(float) == float(plev_pa)]
        elif "plev" in sub.columns:
            sub = sub[sub["plev"].notna()]
            sub = sub[sub["plev"].astype(float) == float(plev_pa)]
        else:
            raise ValueError(f"CSV for {var_name} is missing plev information.")

    if len(sub) == 0:
        raise ValueError(f"No row found in CSV for var={var_name!r}, plev={plev!r}.")
    if len(sub) > 1:
        raise ValueError(f"Multiple rows found in CSV for var={var_name!r}, plev={plev!r}.")

    return sub.iloc[0]


def _plev_to_pa_for_csv(plev) -> int:
    # converts a pressure level entry to Pa for CSV matching
    # accepts values already in Pa (e.g. 30000) or in hPa (e.g. 300)
    p = float(plev)
    if p < 2000:
        return int(round(p * 100.0))
    return int(round(p))


def _csv_bounds_for_wind_speed(cfg, plot_cfg, plev, difference: bool) -> tuple[float, float]:
    """
    reads wind-speed plotting bounds directly from the precomputed wind-speed CSV

    Rules:
    - difference = False:
        raw -> vmin = 0, vmax = raw_max
        99  -> vmin = 0, vmax = p99
        95  -> vmin = 0, vmax = p95

    - difference = True:
        raw -> vmin = raw_min, vmax = raw_max
        99  -> vmin = p01, vmax = p99
        95  -> vmin = p05, vmax = p95
    """
    csv_path = _csv_path_from_plot_cfg(cfg, plot_cfg, difference=difference, background="speed")
    df = pd.read_csv(csv_path)

    row = _select_wind_speed_row(df, plev)
    vmin_col, vmax_col = _csv_percentile_columns(plot_cfg.range_source.percentile, difference=difference)

    if difference:
        vmin = float(row[vmin_col])
    else:
        vmin = 0.0

    vmax = float(row[vmax_col])

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(
            f"Non-finite bounds read from CSV for plev={plev!r}, difference={difference}: "
            f"vmin={vmin}, vmax={vmax}"
        )

    if vmin == vmax:
        eps = max(abs(vmax) * 0.05, 1e-12)
        vmin -= eps
        vmax += eps

    return vmin, vmax


def _csv_bounds_for_regular_var(cfg, plot_cfg, var_name: str, plev, difference: bool) -> tuple[float, float]:
    csv_path = _csv_path_from_plot_cfg(cfg, plot_cfg, difference=difference, background="pressure")
    df = pd.read_csv(csv_path)

    row = _select_regular_row(df, var_name=var_name, plev=plev)
    vmin_col, vmax_col = _csv_percentile_columns(plot_cfg.range_source.percentile, difference=difference)

    if difference:
        vmin = float(row[vmin_col])
    else:
        perc = str(plot_cfg.range_source.percentile).lower()
        if perc == "raw":
            vmin = float(row["raw_min"])
        elif perc == "99":
            vmin = float(row["p01"])
        elif perc == "95":
            vmin = float(row["p05"])
        else:
            raise ValueError(f"Unknown percentile option: {plot_cfg.range_source.percentile}")

    vmax = float(row[vmax_col])

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(
            f"Non-finite bounds read from CSV for var={var_name!r}, plev={plev!r}, difference={difference}: "
            f"vmin={vmin}, vmax={vmax}"
        )

    if vmin == vmax:
        eps = max(abs(vmax) * 0.05, 1e-12)
        vmin -= eps
        vmax += eps

    return vmin, vmax


def _dynamic_bounds(arrays: Iterable[xr.DataArray], percentile=99, difference: bool = False, force_zero_min: bool = False) -> tuple[float, float]:
    vals = []
    for da in arrays:
        data = np.asarray(da.values).ravel()
        data = data[np.isfinite(data)]
        if data.size:
            vals.append(data)

    if not vals:
        raise ValueError("Could not determine colour limits because all arrays are empty or NaN.")

    vals = np.concatenate(vals)
    perc = str(percentile).lower()

    if difference:
        if perc == "99":
            vmin, vmax = np.nanpercentile(vals, [1, 99])
        elif perc == "95":
            vmin, vmax = np.nanpercentile(vals, [5, 95])
        elif perc == "raw":
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        else:
            raise ValueError(f"Unknown percentile option: {percentile}")
    else:
        if force_zero_min:
            vmin = 0.0
            if perc == "99":
                vmax = np.nanpercentile(vals, 99)
            elif perc == "95":
                vmax = np.nanpercentile(vals, 95)
            elif perc == "raw":
                vmax = np.nanmax(vals)
            else:
                raise ValueError(f"Unknown percentile option: {percentile}")
        else:
            if perc == "99":
                vmin, vmax = np.nanpercentile(vals, [1, 99])
            elif perc == "95":
                vmin, vmax = np.nanpercentile(vals, [5, 95])
            elif perc == "raw":
                vmin, vmax = np.nanmin(vals), np.nanmax(vals)
            else:
                raise ValueError(f"Unknown percentile option: {percentile}")

    vmin = float(vmin)
    vmax = float(vmax)

    if vmin == vmax:
        eps = max(abs(vmax) * 0.05, 1e-12)
        vmin -= eps
        vmax += eps

    return vmin, vmax


def _bounds_for_fields(cfg, plot_cfg, arrays: list[xr.DataArray], plev, difference: bool) -> tuple[float, float]:
    manual = _maybe_manual_bounds(plot_cfg)
    if manual is not None:
        return manual
    bg_mode = _background_mode(plot_cfg)
    try:
        if bg_mode == "speed":
            return _csv_bounds_for_wind_speed(cfg, plot_cfg, plev=plev, difference=difference)

        if bg_mode == "pressure":
            return _csv_bounds_for_regular_var(
                cfg,
                plot_cfg,
                var_name="psl",
                plev="surface",
                difference=difference,
            )

        raise ValueError(f"Unsupported background mode: {bg_mode}")

    except Exception as exc:
        warnings.warn(
            f"Falling back to dynamic colour range for background={bg_mode!r}, plev={plev!r}, difference={difference}: {exc}"
        )
        return _dynamic_bounds(
            arrays,
            percentile=plot_cfg.range_source.percentile,
            difference=difference,
            force_zero_min=(bg_mode == "speed" and not difference),
        )


def _subset_background_and_vectors_for_location(plot_cfg, background: xr.DataArray, u: xr.DataArray, v: xr.DataArray):
    return (
        _subset_for_location(background, plot_cfg),
        _subset_for_location(u, plot_cfg),
        _subset_for_location(v, plot_cfg),
    )


def _plot_wind_map(ax, background: xr.DataArray, u: xr.DataArray, v: xr.DataArray, plot_cfg, 
                   vmin: float, vmax: float):
    projection, extent = _projection_and_extent(plot_cfg)
    location = _normalise_location(plot_cfg.location)

    ax.coastlines(linewidth=0.9, color=str(plot_cfg.coastline_colour))
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    if location in POLAR_LOCATIONS:
        theta = np.linspace(0, 2 * np.pi, 200)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="black",
        alpha=0.35,
        linestyle="--",
    )
    # gl.top_labels = False
    # gl.right_labels = False

    if not plot_cfg.difference and plot_cfg.background == "speed":
        cmap = str(plot_cfg.colour_speed)
    elif not plot_cfg.difference and plot_cfg.background == "pressure":
        cmap = str(plot_cfg.colour_pressure)
    elif plot_cfg.difference:
        cmap = str(plot_cfg.colour_diff)

    if location in CYCLIC_POINT_NEEDED:
        background_plot, lon_plot = add_cyclic_point(background.values, coord=background["lon"].values)
        u_plot, _ = add_cyclic_point(u.values, coord=u["lon"].values)
        v_plot, _ = add_cyclic_point(v.values, coord=v["lon"].values)
        lat_plot = background["lat"].values
    else:
        background_plot = background.values
        u_plot = u.values
        v_plot = v.values
        lon_plot = background["lon"].values
        lat_plot = background["lat"].values

    cf = ax.contourf(
        lon_plot,
        lat_plot,
        background_plot,
        levels=np.linspace(vmin, vmax, 21),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extend="both",
        transform=ccrs.PlateCarree(),
    )

    if not bool(plot_cfg.difference):
        skip = int(plot_cfg.skip)
        q = ax.quiver(
            lon_plot[::skip],
            lat_plot[::skip],
            u_plot[::skip, ::skip],
            v_plot[::skip, ::skip],
            transform=ccrs.PlateCarree(),
            pivot="middle",
            scale=int(plot_cfg.scale),
            width=0.002,
        )

        q_ref = plot_cfg.q_ref #5 if plev_is_surface_from_dims(u) else 10
        ax.quiverkey(q, X=0.9, Y=-0.06, U=q_ref, label=f"{q_ref} m s$^{{-1}}$", labelpos="E")

    return cf


def plev_is_surface_from_dims(u: xr.DataArray) -> bool:
    return "plev" not in u.coords and "plev" not in u.dims


def _era5_requested(plot_cfg) -> bool:
    return bool(plot_cfg.map_era5) and not bool(plot_cfg.difference)


def run(cfg):
    plot_cfg = cfg.plots.wind
    bg_mode = _background_mode(plot_cfg)

    if plot_cfg.only_mean and not plot_cfg.include_ensemble_mean_as_member:
        raise ValueError("wind.only_mean=true requires wind.include_ensemble_mean_as_member=true.")

    start, end = resolve_period(cfg, plot_cfg)
    plevs = _plev_entries(plot_cfg)
    figsize = _resolve_figsize(plot_cfg, "map")
    add_dir = str(plot_cfg.special_outdir) if plot_cfg.special_outdir else ""

    if bool(plot_cfg.map_era5) and bool(plot_cfg.difference):
        warnings.warn(
            "Skipping ERA5 wind maps because wind.map_era5=true and wind.difference=true (ERA5 - ERA5 would be zero)."
        )

    outdir = os.path.join(
        hydra.utils.get_original_cwd(),
        cfg.out.dir,
        "wind_map",
        add_dir,
    )
    os.makedirs(outdir, exist_ok=True)

    for plev in plevs:
        era5_background = era5_u = era5_v = None

        try:
            era5_speed, era5_u, era5_v = _prepare_era5_fields(cfg, plot_cfg, plev, start, end)

            if bg_mode == "speed":
                era5_background = era5_speed
            elif bg_mode == "pressure":
                era5_background = _prepare_era5_background_field(cfg, plot_cfg, start, end)
            else:
                raise ValueError(f"Unsupported background mode: {bg_mode}")

        except Exception as exc:
            if bool(plot_cfg.difference) or bool(plot_cfg.map_era5):
                raise
            warnings.warn(f"Could not load ERA5 fields for plev={plev!r}: {exc}")

        if era5_background is not None:
            era5_background, era5_u, era5_v = _subset_background_and_vectors_for_location(
                plot_cfg,
                era5_background,
                era5_u,
                era5_v,
            )

        for model_name in plot_cfg.models:
            model_cfg = cfg.datasets.models[model_name]
            proper_model_name = getattr(model_cfg, "proper_name", model_name)

            member_to_background: dict[str, xr.DataArray] = {}
            member_to_u: dict[str, xr.DataArray] = {}
            member_to_v: dict[str, xr.DataArray] = {}

            for member in cfg.members:
                speed, u, v = _prepare_model_member_fields(
                    model_cfg,
                    cfg,
                    plot_cfg,
                    member,
                    plev,
                    start,
                    end,
                )

                if bg_mode == "speed":
                    background = speed
                elif bg_mode == "pressure":
                    background = _prepare_model_background_field(
                        model_cfg,
                        cfg,
                        plot_cfg,
                        member,
                        start,
                        end,
                    )
                else:
                    raise ValueError(f"Unsupported background mode: {bg_mode}")

                background, u, v = _subset_background_and_vectors_for_location(plot_cfg, background, u, v)

                if bool(plot_cfg.difference):
                    if era5_background is None:
                        raise ValueError("wind.difference=true requires ERA5 background data to be available.")
                    background = background - era5_background

                member_to_background[member] = background
                member_to_u[member] = u
                member_to_v[member] = v

            background_to_plot = _member_mean_mapping(plot_cfg, cfg, member_to_background)
            u_to_plot = _member_mean_mapping(plot_cfg, cfg, member_to_u)
            v_to_plot = _member_mean_mapping(plot_cfg, cfg, member_to_v)

            bounds_arrays = list(background_to_plot.values())
            vmin, vmax = _bounds_for_fields(
                cfg,
                plot_cfg,
                bounds_arrays,
                plev=plev,
                difference=bool(plot_cfg.difference),
            )

            for member, background_da in background_to_plot.items():
                u_da = u_to_plot[member]
                v_da = v_to_plot[member]
                fname = _wind_filename(model_name, member, plev, plot_cfg, start, end, era5=False)
                outfile = os.path.join(outdir, fname)

                if cfg.out.savefig and not should_compute_output(outfile, getattr(cfg.out, "overwrite", "ask")):
                    continue

                projection, _ = _projection_and_extent(plot_cfg)
                fig = plt.figure(figsize=figsize)
                ax = plt.axes(projection=projection)

                loc = _normalise_location(plot_cfg.location)
                if loc is None or loc == "ortho":
                    ax.set_global()

                cf = _plot_wind_map(ax, background_da, u_da, v_da, plot_cfg, vmin, vmax)
                ax.set_title(
                    _format_title(plot_cfg, model_name, proper_model_name, member, plev, start, end, era5=False),
                    pad=12,
                )

                if plev == "surface":
                    plev_label = "surface"
                else:
                    plev_title_str, _ = plev_strings(plev)
                    plev_label = plev_title_str.strip().removeprefix("at ").strip()

                cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.1, shrink=0.85)
                cbar.set_label(_background_cbar_label(plot_cfg, plev_label))

                plt.tight_layout(pad=2.0)
                if cfg.out.savefig:
                    fig.savefig(outfile, dpi=cfg.out.dpi, bbox_inches="tight")
                plt.close(fig)

        if _era5_requested(plot_cfg) and era5_background is not None:
            vmin_era5, vmax_era5 = _bounds_for_fields(
                cfg,
                plot_cfg,
                [era5_background],
                plev=plev,
                difference=False,
            )

            fname = _wind_filename("era5", "", plev, plot_cfg, start, end, era5=True)
            outfile = os.path.join(outdir, fname)

            if not (cfg.out.savefig and not should_compute_output(outfile, getattr(cfg.out, "overwrite", "ask"))):
                projection, _ = _projection_and_extent(plot_cfg)
                fig = plt.figure(figsize=figsize)
                ax = plt.axes(projection=projection)

                loc = _normalise_location(plot_cfg.location)
                if loc is None or loc == "ortho":
                    ax.set_global()

                cf = _plot_wind_map(ax, era5_background, era5_u, era5_v, plot_cfg, vmin_era5, vmax_era5)
                ax.set_title(_format_title(plot_cfg, "era5", "ERA5", "", plev, start, end, era5=True), pad=12)

                if plev == "surface":
                    plev_label = "surface"
                else:
                    plev_title_str, _ = plev_strings(plev)
                    plev_label = plev_title_str.strip().removeprefix("at ").strip()

                cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.1, shrink=0.85)
                cbar.set_label(_background_cbar_label(plot_cfg, plev_label))

                plt.tight_layout(pad=2.0)
                if cfg.out.savefig:
                    fig.savefig(outfile, dpi=cfg.out.dpi, bbox_inches="tight")
                plt.close(fig)
                