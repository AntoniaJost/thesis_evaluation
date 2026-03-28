from __future__ import annotations

import os
import warnings
from typing import Iterable
import pandas as pd
import cartopy.crs as ccrs
import hydra
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point
import matplotlib.path as mpath

from evaluation.general_functions import (
    resolve_period,
    open_model_da,
    open_era5_da,
    ensemble_mean_as_member,
    should_compute_output,
    model_abbrev,
    plev_strings,
    normalise_plevs,
)
from evaluation.metrics.individual_plots import (
    _normalise_location,
    _projection_and_extent,
    _resolve_figsize,
    _subset_for_location,
)

POLAR_LOCATIONS = {"arctic", "antarctic"}
ALLOWED_PLEVS_PA = [5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 85000, 92500, 100000]
ALLOWED_PLEVS_HPA = [int(v / 100) for v in ALLOWED_PLEVS_PA]


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


def _wind_component_names(plev):
    if plev == "surface":
        return "uas", "vas"
    return "ua", "va"


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


def _time_average(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.dims:
        raise ValueError(f"Expected a 'time' dimension, got dims={da.dims}")
    return da.mean(dim="time")


def _compute_scalar_and_vector_climatology(u: xr.DataArray, v: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    speed = np.sqrt(u ** 2 + v ** 2)
    speed_clim = _time_average(speed)
    u_clim = _time_average(u)
    v_clim = _time_average(v)
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

    return _compute_scalar_and_vector_climatology(u, v)


def _prepare_era5_fields(cfg, plot_cfg, plev, start: str, end: str):
    u_var, v_var = _wind_component_names(plev)
    plev_arg = None if plev == "surface" else plev

    u = open_era5_da(cfg, var=u_var, start=start, end=end, plev=plev_arg)
    v = open_era5_da(cfg, var=v_var, start=start, end=end, plev=plev_arg)
    return _compute_scalar_and_vector_climatology(u, v)


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
    return f"wind{plev_tag}_{model_tag}{member_tag}_{loc_tag}{diff_tag}_{start_tag}-{end_tag}.png"


def _format_title(plot_cfg, model_name: str, proper_model_name: str, member: str, plev, start: str, end: str, era5: bool = False) -> str:
    if plev == "surface":
        plev_title_str = ""
        plev_label = "surface"
        plev_hpa = None
    else:
        plev_title_str, _ = plev_strings(plev)
        plev_label = plev_title_str.strip().removeprefix("at ").strip()
        plev_hpa = int(float(plev)) if float(plev) < 2000 else int(float(plev) / 100)

    time_label = f"{start} to {end}"

    if plot_cfg.title:
        return str(plot_cfg.title).format(
            var="wind",
            long_name="Wind speed",
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

    if era5:
        return f"ERA5 wind climatology{' at ' + plev_label if plev != 'surface' else ' at surface'} | {time_label}"
    if plot_cfg.difference:
        return f"Wind speed difference to ERA5{' at ' + plev_label if plev != 'surface' else ' at surface'} | {proper_model_name} {member} | {time_label}"
    return f"Wind climatology{' at ' + plev_label if plev != 'surface' else ' at surface'} | {proper_model_name} {member} | {time_label}"


def _maybe_manual_bounds(plot_cfg):
    cbar_cfg = plot_cfg.colourbar
    if bool(cbar_cfg.manual):
        if cbar_cfg.manual_vmin is None or cbar_cfg.manual_vmax is None:
            raise ValueError("wind.colourbar.manual=true requires both manual_vmin and manual_vmax.")
        return float(cbar_cfg.manual_vmin), float(cbar_cfg.manual_vmax)
    return None


def _csv_path_from_plot_cfg(cfg, plot_cfg, difference: bool) -> str:
    csv_rel = plot_cfg.range_source.csv_file2 if difference else plot_cfg.range_source.csv_file1
    return os.path.join(hydra.utils.get_original_cwd(), str(csv_rel))


def _csv_percentile_columns(percentile, difference: bool) -> tuple[str, str]:
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
        plev_pa = int(plev)
        if "plev_pa" not in sub.columns:
            raise ValueError("Wind-speed bounds CSV is missing required column 'plev_pa'.")
        sub = sub[sub["plev_pa"].notna()]
        sub = sub[sub["plev_pa"].astype(float) == float(plev_pa)]

    if len(sub) == 0:
        raise ValueError(f"No wind_speed row found in CSV for plev={plev!r}.")
    if len(sub) > 1:
        raise ValueError(f"Multiple wind_speed rows found in CSV for plev={plev!r}.")

    return sub.iloc[0]


def _csv_bounds_for_wind_speed(cfg, plot_cfg, plev, difference: bool) -> tuple[float, float]:
    """
    Read wind-speed plotting bounds directly from the precomputed wind-speed CSV.

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
    csv_path = _csv_path_from_plot_cfg(cfg, plot_cfg, difference=difference)
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


def _dynamic_bounds(arrays: Iterable[xr.DataArray], percentile=99, difference: bool = False) -> tuple[float, float]:
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
        vmin = 0.0
        if perc == "99":
            vmax = np.nanpercentile(vals, 99)
        elif perc == "95":
            vmax = np.nanpercentile(vals, 95)
        elif perc == "raw":
            vmax = np.nanmax(vals)
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

    try:
        return _csv_bounds_for_wind_speed(cfg, plot_cfg, plev=plev, difference=difference)
    except Exception as exc:
        warnings.warn(
            f"Falling back to dynamic colour range for wind at plev={plev!r}, difference={difference}: {exc}"
        )
        return _dynamic_bounds(
            arrays,
            percentile=plot_cfg.range_source.percentile,
            difference=difference,
        )


def _subset_fields_for_location(plot_cfg, speed: xr.DataArray, u: xr.DataArray, v: xr.DataArray):
    return (
        _subset_for_location(speed, plot_cfg),
        _subset_for_location(u, plot_cfg),
        _subset_for_location(v, plot_cfg),
    )


def _plot_wind_map(ax, speed: xr.DataArray, u: xr.DataArray, v: xr.DataArray, plot_cfg, vmin: float, vmax: float):
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

    gl = ax.gridlines(draw_labels=(location == "individual"), linewidth=0.5, color="black", alpha=0.35, linestyle="--")
    if location == "individual":
        gl.top_labels = False
        gl.right_labels = False

    cmap = str(plot_cfg.colour_diff if plot_cfg.difference else plot_cfg.colour_scheme)

    if location is None or location in POLAR_LOCATIONS:
        speed_plot, lon_plot = add_cyclic_point(speed.values, coord=speed["lon"].values)
        u_plot, _ = add_cyclic_point(u.values, coord=u["lon"].values)
        v_plot, _ = add_cyclic_point(v.values, coord=v["lon"].values)
        lat_plot = speed["lat"].values
    else:
        speed_plot = speed.values
        u_plot = u.values
        v_plot = v.values
        lon_plot = speed["lon"].values
        lat_plot = speed["lat"].values

    cf = ax.contourf(
        lon_plot,
        lat_plot,
        speed_plot,
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
            scale=200,
            width=0.002,
        )

        q_ref = 5 if plev_is_surface_from_dims(u) else 10
        ax.quiverkey(q, X=0.9, Y=-0.03, U=q_ref, label=f"{q_ref} m s$^{{-1}}$", labelpos="E")
    return cf


def plev_is_surface_from_dims(u: xr.DataArray) -> bool:
    return "plev" not in u.coords and "plev" not in u.dims


def _era5_requested(plot_cfg) -> bool:
    return bool(plot_cfg.map_era5) and not bool(plot_cfg.difference)


def run(cfg):
    plot_cfg = cfg.plots.wind

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
        era5_speed = era5_u = era5_v = None
        try:
            era5_speed, era5_u, era5_v = _prepare_era5_fields(cfg, plot_cfg, plev, start, end)
        except Exception as exc:
            if bool(plot_cfg.difference) or bool(plot_cfg.map_era5):
                raise
            warnings.warn(f"Could not load ERA5 wind fields for plev={plev!r}: {exc}")

        if era5_speed is not None:
            era5_speed, era5_u, era5_v = _subset_fields_for_location(plot_cfg, era5_speed, era5_u, era5_v)

        for model_name in plot_cfg.models:
            model_cfg = cfg.datasets.models[model_name]
            proper_model_name = getattr(model_cfg, "proper_name", model_name)

            member_to_speed: dict[str, xr.DataArray] = {}
            member_to_u: dict[str, xr.DataArray] = {}
            member_to_v: dict[str, xr.DataArray] = {}

            for member in cfg.members:
                speed, u, v = _prepare_model_member_fields(model_cfg, cfg, plot_cfg, member, plev, start, end)
                speed, u, v = _subset_fields_for_location(plot_cfg, speed, u, v)

                if bool(plot_cfg.difference):
                    if era5_speed is None:
                        raise ValueError("wind.difference=true requires ERA5 wind data to be available.")
                    speed = speed - era5_speed

                member_to_speed[member] = speed
                member_to_u[member] = u
                member_to_v[member] = v

            speed_to_plot = _member_mean_mapping(plot_cfg, cfg, member_to_speed)
            u_to_plot = _member_mean_mapping(plot_cfg, cfg, member_to_u)
            v_to_plot = _member_mean_mapping(plot_cfg, cfg, member_to_v)

            bounds_arrays = list(speed_to_plot.values())
            vmin, vmax = _bounds_for_fields(
                cfg,
                plot_cfg,
                bounds_arrays,
                plev=plev,
                difference=bool(plot_cfg.difference),
            )

            for member, speed_da in speed_to_plot.items():
                u_da = u_to_plot[member]
                v_da = v_to_plot[member]
                fname = _wind_filename(model_name, member, plev, plot_cfg, start, end, era5=False)
                outfile = os.path.join(outdir, fname)

                if cfg.out.savefig and not should_compute_output(outfile, getattr(cfg.out, "overwrite", "ask")):
                    continue

                projection, _ = _projection_and_extent(plot_cfg)
                fig = plt.figure(figsize=figsize)
                ax = plt.axes(projection=projection)
                if _normalise_location(plot_cfg.location) is None:
                    ax.set_global()

                cf = _plot_wind_map(ax, speed_da, u_da, v_da, plot_cfg, vmin, vmax)
                ax.set_title(_format_title(plot_cfg, model_name, proper_model_name, member, plev, start, end, era5=False))
                if plev == "surface":
                    plev_label = "surface"
                else:
                    plev_title_str, _ = plev_strings(plev)
                    plev_label = plev_title_str.strip().removeprefix("at ").strip()
                cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, shrink=0.85)
                if plot_cfg.difference:
                    cbar.set_label(f"Wind speed difference at {plev_label} (m s$^{{-1}}$)")
                else:
                    cbar.set_label(f"Mean wind speed at {plev_label} (m s$^{{-1}}$)")

                plt.tight_layout()
                if cfg.out.savefig:
                    fig.savefig(outfile, dpi=cfg.out.dpi, bbox_inches="tight")
                plt.close(fig)

        if _era5_requested(plot_cfg) and era5_speed is not None:
            vmin_era5, vmax_era5 = _bounds_for_fields(
                cfg,
                plot_cfg,
                [era5_speed],
                plev=plev,
                difference=False,
            )
            fname = _wind_filename("era5", "", plev, plot_cfg, start, end, era5=True)
            outfile = os.path.join(outdir, fname)

            if not (cfg.out.savefig and not should_compute_output(outfile, getattr(cfg.out, "overwrite", "ask"))):
                projection, _ = _projection_and_extent(plot_cfg)
                fig = plt.figure(figsize=figsize)
                ax = plt.axes(projection=projection)
                if _normalise_location(plot_cfg.location) is None:
                    ax.set_global()

                cf = _plot_wind_map(ax, era5_speed, era5_u, era5_v, plot_cfg, vmin_era5, vmax_era5)
                ax.set_title(_format_title(plot_cfg, "era5", "ERA5", "", plev, start, end, era5=True))

                if plev == "surface":
                    plev_label = "surface"
                else:
                    plev_title_str, _ = plev_strings(plev)
                    plev_label = plev_title_str.strip().removeprefix("at ").strip()
                cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, shrink=0.85)
                cbar.set_label(f"Mean wind speed at {plev_label} (m s$^{{-1}}$)")

                plt.tight_layout()
                if cfg.out.savefig:
                    fig.savefig(outfile, dpi=cfg.out.dpi, bbox_inches="tight")
                plt.close(fig)
