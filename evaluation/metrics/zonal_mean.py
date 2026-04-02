from __future__ import annotations

import os

import hydra
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import xarray as xr
import warnings
import matplotlib as mpl
import pandas as pd
from omegaconf import ListConfig

from evaluation.general_functions import (
    resolve_period,
    ensure_allowed_var,
    model_abbrev,
    open_model_da_raw,
    open_era5_da_raw,
    conversion_rules,
    should_compute_output,
    ensemble_mean_as_member,
    normalise_list,
    normalise_plevs,
    accept_Pa_and_hPa,
    format_unit_for_plot
)
from evaluation.metrics.individual_plots import (
    _map_levels_and_ticks,
    _get_map_norm,
    _select_bbox,
    _wrap_lon_360,
)

from evaluation.metrics.soi import _lat_slice

TIME_SELECTIONS = {
    "full": None,
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "jan": [1],
    "feb": [2],
    "mar": [3],
    "apr": [4],
    "may": [5],
    "jun": [6],
    "jul": [7],
    "aug": [8],
    "sep": [9],
    "oct": [10],
    "nov": [11],
    "dec": [12],
}


def _selected_time_slices(plot_cfg) -> list[str]:
    """
    accepts:
    - "full", "DJF", "MAM", "JJA", "SON"
    - single months as strings: "jan", ..., "dec"
    - month numbers as int or strings: 1..12
    - list of the above
    """
    raw = getattr(plot_cfg, "season", "full")
    if raw is None:
        return ["full"]
    if isinstance(raw, (list, tuple, ListConfig)):
        vals = list(raw)
    else:
        vals = [raw]
    out = []
    for v in vals:
        if isinstance(v, (int, np.integer)):
            if 1 <= int(v) <= 12:
                month_key = ["jan", "feb", "mar", "apr", "may", "jun",
                             "jul", "aug", "sep", "oct", "nov", "dec"][int(v) - 1]
                out.append(month_key)
                continue
            raise ValueError(f"Invalid month integer for zonal_mean.season: {v}")

        s = str(v).strip()
        s_lower = s.lower()

        if s in {"full", "DJF", "MAM", "JJA", "SON"}:
            out.append(s)
        elif s_lower in TIME_SELECTIONS:
            out.append(s_lower)
        elif s.isdigit() and 1 <= int(s) <= 12:
            month_key = ["jan", "feb", "mar", "apr", "may", "jun",
                         "jul", "aug", "sep", "oct", "nov", "dec"][int(s) - 1]
            out.append(month_key)
        else:
            raise ValueError(
                f"plots.zonal_mean.season contains invalid entry: {v}. "
                "Allowed: full, DJF, MAM, JJA, SON, jan..dec, or 1..12."
            )
    return out


def _time_selection_label(sel: str) -> str:
    mapping = {
        "full": "Full Year",
        "DJF": "DJF",
        "MAM": "MAM",
        "JJA": "JJA",
        "SON": "SON",
        "jan": "January",
        "feb": "February",
        "mar": "March",
        "apr": "April",
        "may": "May",
        "jun": "June",
        "jul": "July",
        "aug": "August",
        "sep": "September",
        "oct": "October",
        "nov": "November",
        "dec": "December",
    }
    return mapping[sel]


def _subset_time_selection(da: xr.DataArray, sel: str) -> xr.DataArray:
    if "time" not in da.dims:
        raise ValueError(f"Expected 'time' dimension, got {da.dims}")

    if sel == "full":
        out = da
    else:
        months = TIME_SELECTIONS[sel]
        out = da.sel(time=da.time.dt.month.isin(months))

    if out.sizes.get("time", 0) == 0:
        raise ValueError(f"No timesteps remain after applying time selection '{sel}'.")
    return out


def _select_requested_plevs(da: xr.DataArray, var: str, requested_plevs, context: str = "") -> xr.DataArray:
    """
    selects requested pressure levels while preserving the plev dimension
    code combination from plevs_for_variable, select_plev_if_needed
    behaviour:
    - no 'plev' in data -> error
    - requested_plevs is None -> return all levels unchanged
    - requested_plevs is scalar/list -> keep those levels only
    """
    if "plev" not in da.dims:
        raise ValueError(
            f"Cannot produce an zonal mean map for a variable that contains no pressure levels (variable: {var})."
        )
    plevs = normalise_plevs(requested_plevs)
    if plevs == [None]:
        return da

    available = np.asarray(da["plev"].values, dtype=float)
    indices = []
    for p in plevs:
        target = accept_Pa_and_hPa(p, available)
        matches = np.where(np.isclose(available, target))[0]
        if len(matches) == 0:
            raise ValueError(
                f"Requested plev={p} for variable '{var}' not found in {context}. "
                f"Available plev values: {[float(v) for v in da['plev'].values]}"
            )
        indices.append(int(matches[0]))

    # isel with a list preserves the plev dimension, even for one selected level
    return da.isel(plev=indices)

def _normalise_region(region) -> str:
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
            "plots.zonal_mean.region must be one of: "
            "global, northern, southern, tropics, arctic, antarctic, individual. "
            f"Got: {region}"
        )
    return aliases[reg]


def _subset_for_region(da: xr.DataArray, plot_cfg) -> xr.DataArray:
    region = _normalise_region(getattr(plot_cfg, "region", "global"))

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
        return da.sel(lat=_lat_slice(da, -23.5, 23.5))

    if region == "arctic":
        min_lat = float(plot_cfg.polar.min_latitude)
        return da.sel(lat=_lat_slice(da, min_lat, 90.0))

    if region == "antarctic":
        max_lat = float(plot_cfg.polar.max_latitude)
        return da.sel(lat=_lat_slice(da, -90.0, max_lat))

    raise ValueError(f"Unsupported region: {region}")


def _region_tag_and_label(plot_cfg) -> tuple[str, str]:
    region = _normalise_region(getattr(plot_cfg, "region", "global"))
    if region != "individual":
        labels = {
            "global": "Global",
            "northern": "Northern Hemisphere",
            "southern": "Southern Hemisphere",
            "tropics": "Tropics",
            "arctic": "Arctic",
            "antarctic": "Antarctic",
        }
        return region, labels[region]
    
    lat0 = float(plot_cfg.individual.lat0)
    lat1 = float(plot_cfg.individual.lat1)
    lon0 = _wrap_lon_360(float(plot_cfg.individual.lon0))
    lon1 = _wrap_lon_360(float(plot_cfg.individual.lon1))

    lat0_tag = f"{lat0:g}"
    lat1_tag = f"{lat1:g}"
    lon0_tag = f"{lon0:g}"
    lon1_tag = f"{lon1:g}"

    return (
        f"box_{lat0_tag}_{lat1_tag}_{lon0_tag}_{lon1_tag}",
        f"Area ({lat0_tag} to {lat1_tag}°, {lon0_tag} to {lon1_tag}°E)",
    )


def _validate_variable(cfg, plot_cfg, var: str, da_sample: xr.DataArray):
    ensure_allowed_var(cfg, var)
    if var not in normalise_list(plot_cfg.accepted_vars):
        raise ValueError(
            f"Variable '{var}' is not allowed for zonal_mean. "
            f"Allowed: {normalise_list(plot_cfg.accepted_vars)}"
        )
    if "plev" not in da_sample.dims:
        raise ValueError(
            f"Variable '{var}' has no 'plev' dimension and therefore cannot be used for a vertical zonal-mean plot."
        )


def _prepare_zonal_mean(
    da: xr.DataArray,
    var: str,
    cfg,
    source: str,
    unit_default: str,
    plot_cfg,
    time_selection: str,
) -> tuple[xr.DataArray, str]:
    """
    converts units, subsets region + time selection, then computes zonal + time mean
    returns [plev, lat]
    """
    da, unit = conversion_rules(var, da, cfg, source, unit_default)

    if "lon" not in da.dims:
        raise ValueError(f"Missing 'lon' dimension: {da.dims}")
    if "time" not in da.dims:
        raise ValueError(f"Missing 'time' dimension: {da.dims}")
    if "plev" not in da.dims:
        raise ValueError(
            f"Missing 'plev' dimension after loading/selection: {da.dims}. "
            "A vertical zonal-mean plot requires pressure levels."
        )

    da = _subset_for_region(da, plot_cfg)
    da = _subset_time_selection(da, time_selection)

    if da.sizes.get("lon", 0) == 0 or da.sizes.get("lat", 0) == 0:
        raise ValueError("No spatial data remain after regional subsetting.")
    if da.sizes.get("time", 0) == 0:
        raise ValueError("No timesteps remain after time selection.")

    da = da.mean("lon")
    da = da.mean("time")
    return da.transpose("plev", "lat"), unit


def _align_era5_to_model_levels(era5: xr.DataArray, model_da: xr.DataArray) -> xr.DataArray:
    """
    reduces ERA5 to the exact plev values of the model data
    """
    if "plev" not in era5.dims or "plev" not in model_da.dims:
        raise ValueError(
            f"Need 'plev' in both ERA5 and model data for alignment. "
            f"ERA5 dims={era5.dims}, model dims={model_da.dims}"
        )
    return era5.sel(plev=model_da["plev"])


def _get_meta(cfg, var: str) -> tuple[str, str]:
    # same as in iter_vars_and_plevs
    if var in cfg.variables.meta:
        meta = cfg.variables.meta[var]
        long_name = meta.long_name
        unit = format_unit_for_plot(meta.unit)
    else:
        long_name = var
        unit = ""
    return long_name, unit


def _get_cmap(plot_cfg) -> str:
    return plot_cfg.cmap_difference if plot_cfg.difference else plot_cfg.cmap_absolute


def _get_zonal_vmin_vmax(cfg, plot_cfg, var: str):
    """
    reads plotting bounds from CSV using range-table logic
    """
    cbar_cfg = plot_cfg.colourbar
    if not getattr(cbar_cfg, "use_csv_ranges", False):
        return None, None
    
    if plot_cfg.difference:
        prefix = "diff_to_era5"
    else:
        prefix = "raw"

    csv_file = os.path.join(
        hydra.utils.get_original_cwd(),
        cbar_cfg.csv_file,
    )
    df = pd.read_csv(csv_file)
    # the following is almost identical to get_range_from_csv, but slightly adapted to the plev csvs
    # for zonal_mean, we always want the combined model/member row
    subset = df[
        (df["variable"] == var)
        & (df["source"] == "models_all_members_combined")
    ]
    if len(subset) == 0:
        warnings.warn(
            f"zonal_mean: no CSV range row found for var='{var}' "
            f"and source='models_all_members_combined' in {csv_file}. "
            "Falling back to automatic colour scaling."
        )
        return None, None

    row = subset.iloc[0]
    percentile = str(cbar_cfg.percentile).lower()
    
    if percentile == "raw":
        vmin = row[f"{prefix}_min"]
        vmax = row[f"{prefix}_max"]
    elif percentile == "99":
        vmin = row[f"{prefix}_p01"]
        vmax = row[f"{prefix}_p99"]
    elif percentile == "95":
        vmin = row[f"{prefix}_p05"]
        vmax = row[f"{prefix}_p95"]
    else:
        raise ValueError(
            f"Unknown colourbar percentile setting: {percentile}. "
            "Expected one of: raw, 99, 95"
        )

    if pd.isna(vmin) or pd.isna(vmax):
        warnings.warn(
            f"zonal_mean: CSV returned NaN bounds for var='{var}'. "
            "Falling back to automatic colour scaling."
        )
        return None, None

    vmin = float(vmin)
    vmax = float(vmax)

    if np.isclose(vmin, vmax):
        warnings.warn(
            f"zonal_mean: degenerate CSV bounds for var='{var}' "
            f"(vmin≈vmax≈{vmin}). Falling back to automatic colour scaling."
        )
        return None, None

    return vmin, vmax


def _plot_panel(ax, da2d: xr.DataArray, title: str, cmap: str, plot_cfg, vmin=None, vmax=None, levels=None, show_ylabel: bool = True):
    plev = da2d["plev"].values

    # convert Pa -> hPa for plotting if needed
    if np.nanmax(plev) > 2000:
        plev = plev / 100.0

    norm = None
    if vmin is not None and vmax is not None:
        norm = _get_map_norm(plot_cfg, vmin, vmax)

    cf = ax.contourf(
        da2d["lat"].values,
        plev,
        da2d.values,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
    )
    # cf = ax.pcolormesh(
    #     da2d["lat"].values,
    #     plev,
    #     da2d.values,
    #     cmap=cmap,
    #     shading="nearest"
    # )

    ax.set_title(title, pad=12)
    ax.set_xlabel("Latitude")
    if show_ylabel:
        ax.set_ylabel("Pressure (hPa)")
    else:
        ax.set_ylabel("")
    if getattr(plot_cfg, "ylog", True):
        ax.set_yscale("log")
    if getattr(plot_cfg, "invert_yaxis", True):
        ax.invert_yaxis()

    major_ticks = [50, 100, 150, 200, 300, 500, 700, 850, 1000]
    minor_ticks = [250, 400, 600, 925, 1000]

    ymin, ymax = ax.get_ylim()
    low, high = min(ymin, ymax), max(ymin, ymax)
    minor_ticks = [p for p in minor_ticks if low <= p <= high]
    major_ticks = [p for p in major_ticks if low <= p <= high]

    ax.set_yticks(major_ticks)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticklabels([str(p) for p in major_ticks])
    ax.set_yticks(minor_ticks, minor=True)

    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])

    return cf


def _make_suptitle(plot_cfg, long_name: str, unit: str, start: str, end: str, time_selection: str, 
                   region_label: str) -> str:    
    # use config title if provided, otherwise build a default title
    # similar to _format_title
    ref_type = getattr(plot_cfg.reference, "type", None) if hasattr(plot_cfg, "reference") else None
    ref_model = getattr(plot_cfg.reference, "model", None) if hasattr(plot_cfg, "reference") else None
    time_label = _time_selection_label(time_selection)

    if getattr(plot_cfg, "title", None):
        return plot_cfg.title.format(
            long_name=long_name,
            unit=unit,
            start=start,
            end=end,
            difference=plot_cfg.difference,
            reference_type=ref_type,
            reference_model=ref_model,
            season=time_label,
            region=region_label,
        )

    if plot_cfg.difference:
        ref_str = "ERA5" if ref_type == "era5" else str(ref_model)
        return f"{time_label} zonal mean {long_name} difference ({unit})\nModel - {ref_str} | {region_label}\n{start} to {end}"

    return f"{time_label} zonal mean {long_name} ({unit})\n{region_label}\n{start} to {end}"


def _resolve_figsize(plot_cfg, n_panels: int) -> tuple[float, float]:
    if getattr(plot_cfg, "figsize", None):
        return tuple(plot_cfg.figsize)
    width_per_panel = 7
    base_height = 7
    max_width = 30 # prevent extremely wide figures
    width = min(max_width, max(4.2, n_panels * width_per_panel))
    return (width, base_height)


def _plot_single_panel_figure(da2d, title, long_name, unit, plot_cfg, start, end, time_selection, 
                              region_label, vmin=None, vmax=None, levels=None, ticks=None):
    figsize = _resolve_figsize(plot_cfg, 1)
    fig, ax = plt.subplots(figsize=figsize)

    cmap = _get_cmap(plot_cfg)
    mappable = _plot_panel(ax, da2d, title, cmap, plot_cfg, vmin=vmin, vmax=vmax, levels=levels)

    fig.suptitle(_make_suptitle(plot_cfg, long_name, unit, start, end, time_selection, region_label), y=0.96)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    cbar = fig.colorbar(mappable, ax=ax, orientation="vertical", shrink=0.9)
    if ticks is not None:
        cbar.set_ticks(ticks)
    cbar.set_label(f"{long_name} ({unit})")

    return fig


def _plot_panel_row(panels, long_name, unit, plot_cfg, start, end, time_selection, region_label,
                     vmin=None, vmax=None, levels=None, ticks=None):
    n_panels = len(panels)
    figsize = _resolve_figsize(plot_cfg, n_panels)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_panels,
        figsize=figsize,
        squeeze=False,
    )

    axes_flat = axes.flatten()
    cmap = _get_cmap(plot_cfg)

    mappable = None
    for i, (ax, (title, da2d)) in enumerate(zip(axes_flat, panels)):
        mappable = _plot_panel(ax, da2d, title, cmap, plot_cfg, vmin=vmin, vmax=vmax, levels=levels, show_ylabel=(i==0))

    fig.suptitle(_make_suptitle(plot_cfg, long_name, unit, start, end, time_selection, region_label), y=0.96, x=0.43)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if mappable is not None:
        cbar = fig.colorbar(
            mappable,
            ax=axes_flat,
            orientation="vertical",
            shrink=0.9,
            pad=0.02,
        )
        if ticks is not None:
            cbar.set_ticks(ticks)
        cbar.set_label(f"{long_name} ({unit})")

    return fig


def _save_or_show(fig, cfg, outfile: str):
    if cfg.out.savefig:
        fig.savefig(outfile, dpi=cfg.out.dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def run(cfg):
    plot_cfg = cfg.plots.zonal_mean

    if len(plot_cfg.models) == 0:
        raise ValueError("zonal_mean requires at least one model")

    start, end = resolve_period(cfg, plot_cfg)
    vars_to_plot = normalise_list(plot_cfg.variable)
    requested_plevs = plot_cfg.plev if hasattr(plot_cfg, "plev") else None
    time_selections = _selected_time_slices(plot_cfg)
    region_tag, region_label = _region_tag_and_label(plot_cfg)
    # decide once whether an ERA5 panel should actually be plotted
    show_era5_panel = getattr(plot_cfg, "map_era5", False)
    if plot_cfg.difference and show_era5_panel:
        warnings.warn(
            "zonal_mean: map_era5=true is ignored when difference=true. "
            "ERA5 is used only as a reference and is not plotted as its own panel."
        )
        show_era5_panel = False

    # we still may need ERA5 as a reference in difference mode
    need_era5_data = show_era5_panel or (
        plot_cfg.difference and plot_cfg.reference.type == "era5"
    )

    add_dir = str(str(plot_cfg.special_outdir) if plot_cfg.special_outdir else "")
    outdir = os.path.join(
        hydra.utils.get_original_cwd(),
        cfg.out.dir,
        "zonal_mean",
        add_dir
    )
    os.makedirs(outdir, exist_ok=True)

    diff_tag = "diff" if plot_cfg.difference else "abs"
    if plot_cfg.freq == "monthly":
        start_tag = start[:7].replace("-", "")
        end_tag = end[:7].replace("-", "")
    else:
        start_tag = start.replace("-", "")
        end_tag = end.replace("-", "")

    for var in vars_to_plot:
        long_name, unit_default = _get_meta(cfg, var)

        for time_selection in time_selections:
            time_sel_tag = time_selection
            if plot_cfg.colourbar.manual_vmin is not None and plot_cfg.colourbar.manual_vmax is not None:
                vmin = float(plot_cfg.colourbar.manual_vmin)
                vmax = float(plot_cfg.colourbar.manual_vmax)
            else:
                vmin, vmax = _get_zonal_vmin_vmax(cfg, plot_cfg, var)

            if vmin is not None and vmax is not None:
                if plot_cfg.colourbar.use_custom_bins:
                    levels, ticks = _map_levels_and_ticks(vmin, vmax, plot_cfg)
                else:
                    levels = np.linspace(vmin, vmax, 21)
                    ticks = None
            else:
                levels = None
                ticks = None

            # load ERA5 once per variable if it is needed either as reference or as panel
            era5_raw = None
            if need_era5_data:
                era5_raw = open_era5_da_raw(cfg, var, start, end)
                era5_raw = _select_requested_plevs(
                    era5_raw,
                    var=var,
                    requested_plevs=requested_plevs,
                    context="ERA5",
                )

            for model_name in plot_cfg.models:
                model_cfg = cfg.datasets.models[model_name]
                model_tag = model_abbrev(model_name)

                # decide output names early, before computation
                if plot_cfg.all_single_plots:
                    panel_tags = []

                    if show_era5_panel:
                        panel_tags.append("era5")

                    if plot_cfg.only_mean:
                        panel_tags.append("mean")
                    else:
                        panel_tags.extend(cfg.members)
                        if getattr(plot_cfg, "include_ensemble_mean_as_member", False):
                            panel_tags.append("mean")

                    outfiles = {
                        tag: os.path.join(
                            outdir,
                            f"{var}_{model_tag}_{tag}_{region_tag}_{time_sel_tag}_{diff_tag}_{start_tag}-{end_tag}.png",
                        )
                        for tag in panel_tags
                    }

                    compute_flags = {}
                    any_to_compute = False
                    if cfg.out.savefig:
                        for tag, outfile in outfiles.items():
                            do_compute = should_compute_output(
                                outfile, getattr(cfg.out, "overwrite", "ask")
                            )
                            compute_flags[tag] = do_compute
                            if do_compute:
                                any_to_compute = True
                    else:
                        for tag in outfiles:
                            compute_flags[tag] = True
                        any_to_compute = True

                    if not any_to_compute:
                        continue

                else:
                    row_outfile = os.path.join(
                        outdir,
                        f"{var}_{model_tag}_{region_tag}_{time_sel_tag}_{diff_tag}_{start_tag}-{end_tag}.png",
                    )

                    if cfg.out.savefig and not should_compute_output(
                        row_outfile, getattr(cfg.out, "overwrite", "ask")
                    ):
                        continue

                # load and process all members for this model
                member_to_da = {}
                first_model_raw = None
                unit = unit_default

                for i, member in enumerate(cfg.members):
                    model_raw = open_model_da_raw(
                        model_cfg,
                        cfg,
                        member,
                        var,
                        model_cfg.modelname,
                        plot_cfg.freq,
                        start,
                        end,
                        grid=plot_cfg.grid,
                    )

                    # validate structure once per model
                    if i == 0:
                        _validate_variable(cfg, plot_cfg, var, model_raw)

                    model_raw = _select_requested_plevs(
                        model_raw,
                        var=var,
                        requested_plevs=requested_plevs,
                        context=f"{model_name}, {member}",
                    )

                    if first_model_raw is None:
                        first_model_raw = model_raw

                    model_2d, unit = _prepare_zonal_mean(
                        da=model_raw,
                        var=var,
                        cfg=cfg,
                        source="model",
                        unit_default=unit_default,
                        plot_cfg=plot_cfg,
                        time_selection=time_selection
                    )
                    if plot_cfg.difference:
                        if plot_cfg.reference.type == "era5":
                            ref_raw = _align_era5_to_model_levels(era5_raw, model_raw)

                        elif plot_cfg.reference.type == "model":
                            ref_cfg = cfg.datasets.models[plot_cfg.reference.model]

                            ref_raw = open_model_da_raw(
                                ref_cfg,
                                cfg,
                                member,
                                var,
                                ref_cfg.modelname,
                                plot_cfg.freq,
                                start,
                                end,
                                grid=plot_cfg.grid,
                            )

                            ref_raw = _select_requested_plevs(
                                ref_raw,
                                var=var,
                                requested_plevs=requested_plevs,
                                context=f"reference model {plot_cfg.reference.model}, {member}",
                            )
                        else:
                            raise ValueError("reference.type must be 'era5' or 'model'")

                        ref_source = "era5" if plot_cfg.reference.type == "era5" else "model"
                        ref_2d, _ = _prepare_zonal_mean(
                            ref_raw,
                            var,
                            cfg,
                            ref_source,
                            unit_default,
                            plot_cfg,
                            time_selection
                        )
                        model_2d = model_2d - ref_2d

                    member_to_da[member] = model_2d

                if getattr(plot_cfg, "include_ensemble_mean_as_member", False):
                    member_to_da = ensemble_mean_as_member(member_to_da, name="mean")

                # build panels for this one model
                panels = []

                if show_era5_panel:
                    era5_for_plot = (
                        _align_era5_to_model_levels(era5_raw, first_model_raw)
                        if plot_cfg.reduce_era5_to_same_levels
                        else era5_raw
                    )

                    era5_2d, unit = _prepare_zonal_mean(
                        era5_for_plot,
                        var,
                        cfg,
                        "era5",
                        unit_default,
                        plot_cfg,
                        time_selection
                    )

                    panels.append(("era5", "ERA5", era5_2d))

                if plot_cfg.only_mean:
                    if "mean" not in member_to_da:
                        raise KeyError(
                            "only_mean=True requires include_ensemble_mean_as_member=True "
                            "so that a mean panel exists."
                        )
                    panels.append(("mean", model_cfg.proper_name, member_to_da["mean"]))
                else:
                    for member in cfg.members:
                        panels.append(
                            (
                                member,
                                f"{model_cfg.proper_name} ({member})",
                                member_to_da[member],
                            )
                        )

                    if getattr(plot_cfg, "include_ensemble_mean_as_member", False):
                        panels.append(
                            (
                                "mean",
                                f"{model_cfg.proper_name} (mean)",
                                member_to_da["mean"],
                            )
                        )

                # now either save one figure per panel, or one row for this model
                if plot_cfg.all_single_plots:
                    for tag, title, da2d in panels:
                        if cfg.out.savefig and not compute_flags[tag]:
                            continue

                        fig = _plot_single_panel_figure(
                            da2d,
                            title,
                            long_name,
                            unit,
                            plot_cfg,
                            start,
                            end,
                            vmin=vmin,
                            vmax=vmax,
                            levels=levels,
                            ticks=ticks,
                        )

                        _save_or_show(fig, cfg, outfiles[tag])

                else:
                    fig = _plot_panel_row(
                        [(title, da2d) for _, title, da2d in panels],
                        long_name,
                        unit,
                        plot_cfg,
                        start,
                        end,
                        time_selection,
                        region_label,
                        vmin=vmin,
                        vmax=vmax,
                        levels=levels,
                        ticks=ticks,
                    )

                    _save_or_show(fig, cfg, row_outfile)
