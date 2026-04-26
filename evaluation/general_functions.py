# evaluation/general_functions.py
from __future__ import annotations
from omegaconf import ListConfig
import glob
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import xarray as xr
import pandas as pd
import os
from functools import lru_cache
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

def normalise_list(value):
    """
    convert string or list-like config entry into a list
    """
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]


def model_abbrev(name: str) -> str:
    return {
        "forced_sst": "sst0K",
        "forced_sst_2k": "sst2K",
        "forced_sst_4k": "sst4K",
        "free_run_control": "FRc",
        "free_run_prediction": "FR15",
    }.get(name, name)


def resolve_period(cfg, plot_cfg) -> Tuple[str, str]:
    """
    Priority:
      1) plot_cfg.time.use_named -> cfg.periods.named[...]
      2) plot_cfg.time.start/end (both not None)
      3) cfg.periods.default
    """
    use_named = getattr(plot_cfg.time, "use_named", None)
    if use_named:
        p = cfg.periods.named[use_named] # look up how those periods are defined
        return p.start, p.end

    start = getattr(plot_cfg.time, "start", None)
    end = getattr(plot_cfg.time, "end", None)
    if start and end:
        return start, end

    return cfg.periods.default.start, cfg.periods.default.end


def open_single_match(pattern: str) -> str:
    """
    Finds the files that match "..._aimip_r*i1p1f1_gn_*.nc"
    """
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No files matched pattern:\n{pattern}")
    # if len(matches) > 1:
    #     # keep deterministic: pick lexicographically smallest
    #     matches = sorted(matches)
    return matches[0]


def model_file_pattern(model_cfg, cfg, member: str, var: str, modelname: str, freq: str, grid: str = "gn") -> str:
    """
    expected structure:
      {root}/{member}/{Amon|day}/{var}/{grid}/{filename}
    
    model_cfg is cfg.datasets.models[model_name]
    member is "for m in cfg.members"
    freq, var, modelname & grid come from cfg.plots.*.variable / freq / grid
    """
    if freq not in ("monthly", "daily"):
        raise ValueError(f"freq must be 'monthly' or 'daily', got: {freq}")

    table = cfg.freq_map[freq]  # mapping of monthly=Amon or daily=day
    if freq == "monthly":
        pat = cfg.pattern.monthly.format(var=var, modelname=modelname) # .format makes it a string
    elif freq == "daily":
        pat = cfg.pattern.daily.format(var=var, modelname=modelname)

    return f"{model_cfg.root}/{member}/{table}/{var}/{grid}/{pat}" # full path


def era5_daily_file_pattern(cfg, var: str, grid: str = "gn") -> str:
    """
    expected daily ERA5 CMOR structure:
      {root}/{var}/{grid}/{var}_day_era5_aimip_ERA5_{grid}_*.nc
    """
    root = cfg.datasets.era5.daily.root
    pattern = cfg.datasets.era5.daily.pattern
    return f"{root}/{var}/{grid}/{pattern.format(var=var, grid=grid)}"


def normalise_vars(var_cfg) -> List:
    """
    normalise plot_cfg.variable to a list
    e.g. "tas" -> ["tas"], ["tas", "siconc"] -> ["tas", "siconc"]
    """
    if isinstance(var_cfg, (list, tuple, ListConfig)):
        return list(var_cfg)
    return [var_cfg]


def variable_requires_plev(da: xr.DataArray) -> bool:
    return "plev" in da.dims


def normalise_plevs(plev_cfg) -> List:
    """
    normalise plot_cfg.plev to a list
    e.g.: None -> [None], 50000 -> [50000], [85000, 50000] -> [85000, 50000]
    """
    if plev_cfg is None:
        return [None]
    if isinstance(plev_cfg, (list, tuple, ListConfig)):
        return list(plev_cfg)
    return [plev_cfg]


def accept_Pa_and_hPa(target, available_plevs) -> float:
    """
    accept either Pa or hPa
    if the file uses Pa and the user passes 500, interpret it as 500 hPa -> 50000 Pa.
    """
    target = float(target)
    avail = np.asarray(available_plevs, dtype=float)

    if np.nanmax(avail) > 2000 and target < 2000:
        target = target * 100.0

    return target

def select_plev_if_needed(da: xr.DataArray, var: str, plev=None, context: str = "") -> xr.DataArray:
    """
    if da has a pressure level dimension, select the requested plev
    raises an error if plev is required but not provided
    """
    if "plev" not in da.dims:
        return da

    if plev is None:
        available = [float(v) for v in da["plev"].values]
        raise ValueError(
            f"Variable '{var}' in {context} has a 'plev' dimension, but no plev was provided. "
            f"Please set plots.global_mean.plev. Available plev values: {available}"
        )

    target = accept_Pa_and_hPa(plev, da["plev"].values)
    available = np.asarray(da["plev"].values, dtype=float)

    matches = np.where(np.isclose(available, target))[0]
    if len(matches) == 0:
        raise ValueError(
            f"Requested plev={plev} for variable '{var}' not found in {context}. "
            f"Available plev values: {[float(v) for v in da['plev'].values]}"
        )

    return da.isel(plev=int(matches[0]))


def plevs_for_variable(da: xr.DataArray, requested_plevs) -> List:
    """
    returns plev list to iterate over for this var
    - no plev dimension -> [None]
    - has plev dimension -> requested_plevs must be provided
    """
    if "plev" not in da.dims:
        return [None]

    plevs = normalise_plevs(requested_plevs)
    if plevs == [None]:
        raise ValueError(
            "A variable with dimension 'plev' was selected, but no pressure level was provided. "
            "Please set plots.<plotname>.plev, e.g. plots.global_mean.plev=500 "
            "or something like plots.global_mean.plev='[100,250,500]'."
        )
    return plevs


def open_model_da_raw(model_cfg, cfg, member: str, var: str, modelname: str, freq: str, start: str, end: str,
                      grid: str = "gn") -> xr.DataArray:
    pattern = model_file_pattern(model_cfg, cfg, member=member, var=var, modelname=modelname, freq=freq, grid=grid)
    path = open_single_match(pattern)
    ds = xr.open_dataset(path).sel(time=slice(start, end))
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in {path}. Available: {list(ds.data_vars)}")
    return ds[var]


def open_model_da(model_cfg, cfg, member: str, var: str, modelname: str, freq: str, start: str, end: str, grid: str = "gn", plev=None) -> xr.DataArray:
    """
    Opens (xr.open_dataset) single file for given timeframe
    """
    pattern = model_file_pattern(model_cfg, cfg, member=member, var=var, modelname=modelname, freq=freq, grid=grid)
    path = open_single_match(pattern)
    ds = xr.open_dataset(path).sel(time=slice(start, end))
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in {path}. Available: {list(ds.data_vars)}")
    da = ds[var]
    da = select_plev_if_needed(da, var=var, plev=plev, context=path)
    return da


def open_era5_da_raw(cfg, var: str, start: str, end: str, freq: str = "monthly", grid: str = "gn") -> xr.DataArray:
    """
    Open ERA5 variable without selecting a pressure level.
    needed for zonal_mean
    """
    if var not in cfg.variables.era5_name:
        raise KeyError(
            f"No ERA5 name mapping for var='{var}'. "
            f"Available mappings: {list(cfg.variables.era5_name.keys())}"
        )

    if freq == "monthly":
        era5_var = cfg.variables.era5_name[var] 
    elif freq == "daily":
        era5_var = var

    if freq == "monthly": 
        root = cfg.datasets.era5.monthly.root
        pattern = cfg.datasets.era5.monthly.pattern
        file_var = "ci" if var == "siconc" else var
        path = f"{root}/{pattern.format(var=file_var)}"
    elif freq == "daily":
        path = open_single_match(era5_daily_file_pattern(cfg, var=var, grid=grid))        
    else:
        raise ValueError(f"Unsupported freq for ERA5 loading: {freq}. Expected 'monthly' or 'daily'.")

    ds = xr.open_dataset(path).sel(time=slice(start, end))

    if era5_var not in ds:
        raise KeyError(
            f"ERA5 variable '{era5_var}' not found in {path}. "
            f"Available: {list(ds.data_vars)}"
        )

    return ds[era5_var]


def open_era5_da(cfg, var: str, start: str, end: str, plev=None,  freq: str = "monthly", grid: str = "gn") -> xr.DataArray:
    # map to ERA5 variable naming
    # if var not in cfg.variables.era5_name:
    #     raise KeyError(
    #         f"No ERA5 name mapping for var='{var}'. "
    #         f"Available mappings: {list(cfg.variables.era5_name.keys())}"
    #     )
    # era5_var = cfg.variables.era5_name[var]
    # root = cfg.datasets.era5.root
    # pattern = cfg.datasets.era5.pattern
    # file_var = "ci" if var == "siconc" else var # handle case of sea ice conc., which uses other variable from era5 (ci, instead of siconc)
    # path = f"{root}/{pattern.format(var=file_var)}"

    # ds = xr.open_dataset(path).sel(time=slice(start, end))
    # if era5_var not in ds:
    #     raise KeyError(
    #         f"ERA5 variable '{era5_var}' not found in {path}. "
    #         f"Available: {list(ds.data_vars)}"
    #     )
    # da = ds[era5_var]
    da = open_era5_da_raw(cfg, var=var, start=start, end=end, freq=freq, grid=grid)
    context = f"ERA5 ({freq})"
    da = select_plev_if_needed(da, var=var, plev=plev, context=context)
    return da


def conversion_rules(var: str, da: xr.DataArray, cfg, source: str, unit_default: str = "") -> tuple[xr.DataArray, str]:
    """
    convert units if entry in config.yaml available, updates unit string
    for daily era5 data, which is cmorised, the same conversions are applied as for the model data
    K <-> °C, scalar to percent
    needed for:
      - ta & tas
      - tos from ERA5
      - ci from ERA5
    """
    rule = cfg.conversions.get(var, None) if hasattr(cfg, "conversions") else None
    if rule is None:
        return da, unit_default

    # make sure tos only gets converted for era5
    applies_to = getattr(rule, "applies_to", "both") # default "both"
    # case tos_ERA5: applies_to = "era5" and source = "era5" -> 'era5' != "both" (TRUE) and 'era5' !='era5' (FALSE) -> FALSE (continues)
    # case tos_MODEL: applies_to = "era5" and source = "model" -> 'era5' != "both" (TRUE) and 'era5' !='model' (TRUE) -> TRUE (returns da)
    # case tas_ERA5: applies_to = 'both' and source='model' -> 'both' != 'both' (FALSE) and 'both' !='era5' (TRUE) -> FALSE (continues)
    # case tas_MODEL: applies_to = 'both' and source='era5' -> 'both' != 'both' (FALSE) and 'both' !='model' (TRUE) -> FALSE (continues)
    valid_sources = {"model", "era5_natural", "era5_cmor"}
    if source not in valid_sources:
        raise ValueError(f"Unknown source={source!r}. Expected one of {sorted(valid_sources)}")
    apply_rule = False
    if applies_to == "both":
        apply_rule = True
    elif applies_to == "model" and source == "model":
        apply_rule = True
    elif applies_to == "era5" and source in {"era5_natural", "era5_cmor"}:
        apply_rule = True
    elif applies_to == "era5_natural" and source == "era5_natural":
        apply_rule = True
    elif applies_to == "era5_cmor" and source == "era5_cmor":
        apply_rule = True

    if not apply_rule:
        return da, unit_default
    
    unit = getattr(rule, "unit", unit_default)
    
    op = rule.op
    val = float(rule.value)
    if op == "sub":
        return da - val, unit
    if op == "add":
        return da + val, unit
    if op == "mul":
        return da * val, unit
    if op == "div":
        return da / val, unit
    # if op == "to_geopot_height":
    #     return da / val, unit
    raise ValueError(f"Unknown conversion op: {op}")


def format_unit_for_plot(unit: str) -> str:
    if unit is None:
        return ""
    unit = str(unit).strip()
    replacements = {
        "m s-1": r"m s$^{-1}$",
        "Pa s-1": r"Pa s$^{-1}$",
        "kg kg-1": r"kg kg$^{-1}$",
        "kg m-2 s-1": r"kg m$^{-2}$ s$^{-1}$",
        "W m-2": r"W m$^{-2}$",
        "m2 s-2": r"m$^{2}$ s$^{-2}$",
    }
    return replacements.get(unit, unit)


def ensemble_mean_as_member(member_to_da: Dict[str, xr.DataArray], name: str = "mean") -> Dict[str, xr.DataArray]:
    """
    compute mean across members and add as an extra "member"
    assumes all DAs are aligned
    """
    keys = sorted(member_to_da.keys())
    da_members = xr.concat([member_to_da[k] for k in keys], dim="member").assign_coords(member=keys)
    out = dict(member_to_da)
    out[name] = da_members.mean("member")
    return out


def ensure_allowed_var(cfg, var: str):
    allowed = list(cfg.variables.allowed)
    if var not in allowed:
        raise ValueError(f"Variable '{var}' not in allowed list: {allowed}")


def normalise_overwrite_mode(mode) -> str:
    """
    normalise overwrite mode from config
    allowed only these values: 'ask', 'true', 'false'
    """
    if isinstance(mode, bool):
        return "true" if mode else "false"

    if mode is None:
        return "ask"

    mode = str(mode).strip().lower()
    if mode in ("ask", "true", "false"):
        return mode

    raise ValueError(
        f"Unknown cfg.out.overwrite value: {mode}. "
        f"Use one of: ask, true, false"
    )


def should_compute_output(outfile: str, overwrite_mode) -> bool:
    """
    decide whether computation should proceed for a target output file
    - true  -> always recompute / overwrite
    - false -> skip if file exists
    - ask   -> prompt user if file exists
    """
    mode = normalise_overwrite_mode(overwrite_mode)
    outfile_name = os.path.basename(outfile)

    if not os.path.exists(outfile):
        return True

    if mode == "true":
        return True

    if mode == "false":
        print(f"Skipping existing file: {outfile}")
        return False

    answer = input(f"{outfile} already exists. Recompute and overwrite? [y/n]: ").strip().lower()
    if answer in ("y", "yes"):
        return True

    print(f"Skipping existing file: {outfile_name}")
    return False


def iter_vars_and_plevs(cfg, plot_cfg):
    """
    yields per-variable plotting information: var, long_name, unit, plevs, start, end
    (yield pauses the function and returns one value at a time)
    """
    start, end = resolve_period(cfg, plot_cfg)
    vars_to_plot = normalise_vars(plot_cfg.variable)
    requested_plevs = getattr(plot_cfg, "plev", None)

    sample_model_name = plot_cfg.models[0]
    sample_model_cfg = cfg.datasets.models[sample_model_name]
    sample_member = cfg.members[0]

    for var in vars_to_plot:
        ensure_allowed_var(cfg, var)

        meta = cfg.variables.meta.get(var, None)
        long_name = meta.long_name if meta else var
        unit = meta.unit if meta else ""

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

        yield {
            "var": var,
            "long_name": long_name,
            "unit": unit,
            "plevs": plevs,
            "start": start,
            "end": end,
        }


def plev_strings(plev):
    """
    return title/file strings for a pressure level
    """
    if plev is None:
        return "", ""

    plev_pa = int(float(plev) * 100) if float(plev) < 2000 else int(float(plev))
    plev_hpa = int(plev_pa / 100)
    return f" at {plev_hpa} hPa", f"@{plev_hpa}hPa"


@lru_cache
def load_range_table(path): # caches csv
    return pd.read_csv(path)


def convert_bounds_with_rules(
    vmin: float,
    vmax: float,
    var: str,
    cfg,
    source: str = "model",
    unit_default: str = "",
) -> tuple[float, float]:
    """
    apply the same conversion_rules used for data to plotting bounds
    for the case when CSV ranges were computed in native units but plots use converted units
    """
    if var in {"ta", "tas"}:
        return float(vmin), float(vmax)
    dummy = xr.DataArray([float(vmin), float(vmax)])
    dummy_conv, _ = conversion_rules(var, dummy, cfg, source, unit_default)
    return float(dummy_conv.values[0]), float(dummy_conv.values[1])


def get_range_from_csv(percentile, csv_file: str, var: str, plev: int | None, prefix: str = "slope"):
    """
    reads min/max plotting range from CSV for a given variable and pressure level
    returns vmin, vmax
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Required CSV file for computing the bias maps not found:\n{csv_file}\n"
            "Run the range_summary script first (python -m evaluation.range_summary) to generate it and ensure that bias_map.yaml receives the correct path."
        )

    df = load_range_table(csv_file)

    # detect variable column name
    var_col = "variable" if "variable" in df.columns else "var"
    df = df[df[var_col] == var]

    # only filter by pressure level if one is requested
    if plev is not None:
        if "plev_pa" not in df.columns or "plev_hpa" not in df.columns:
            raise ValueError(
                f"Pressure-level variable requested (plev={plev}), but CSV does not contain "
                f"'plev_pa' and 'plev_hpa'. Available columns: {list(df.columns)}"
            )

        # decide whether user input is Pa or hPa; if available values are in Pa and input < 2000, interpret as hPa
        plev_pa = accept_Pa_and_hPa(plev, df["plev_pa"].dropna().values)

        if float(plev) < 2000:
            # user likely gave hPa
            df = df[df["plev_hpa"] == float(plev)]
        else:
            # user gave Pa
            df = df[df["plev_pa"] == plev_pa]

    if df.empty:
        raise ValueError(f"No range info found in CSV for {var} at plev={plev}")

    row = df.iloc[0]
    percentile = str(percentile).lower()

    if percentile == "99":
        vmin = row[f"{prefix}_p01"]
        vmax = row[f"{prefix}_p99"]
    elif percentile == "95":
        vmin = row[f"{prefix}_p05"]
        vmax = row[f"{prefix}_p95"]
    elif percentile == "raw":
        vmin = row[f"{prefix}_min"]
        vmax = row[f"{prefix}_max"]
    else:
        raise ValueError(f"Unknown percentile option: {percentile}")

    return float(vmin), float(vmax)


def detrend_dataarray(da: xr.DataArray, dim: str = "time", start: str = "", end: str = "", preserve_mean: bool = True) -> xr.DataArray:
    """
    - removes only the linear slope along `dim`
    - if `start` and `end` are given, the linear fit is estimated on that subset, but the resulting slope correction is applied to the full DataArray
    Behaviour:
    - preserve_mean=True:
        remove only the varying part of the fitted line -> keep the mean level of the fitted period
    - preserve_mean=False:
        remove the full fitted line (slope+intercept)
    """
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray dims {da.dims}")
    # enable option to use only a subset of the (time) period to detrend
    if start and end:
        fit_da = da.sel({dim: slice(start, end)})
        if fit_da.sizes[dim] == 0:
            raise ValueError(
                f"No data found in base_period {start} - {end} for detrending."
            )
    else:  # if no period is given, the full dataset is used
        fit_da = da
    # polyfit along specified dimension
    polyfit = fit_da.polyfit(dim=dim, deg=1, skipna=True) #deg=1=linear trend
    coeffs = polyfit.polyfit_coefficients # extract slope and intercept
    trend_full = xr.polyval(da[dim], coeffs) # fitted trend for entire dataset

    if preserve_mean:
        # remove only the varying part of the fitted line -> this keeps the mean level of the fitted period
        trend_ref = xr.polyval(fit_da[dim], coeffs).mean(dim=dim)
        detrended = da - (trend_full - trend_ref)
    else:
        # old behaviour: remove the full fitted line
        detrended = da - trend_full

    return detrended


def custom_colour(name: str):
    """
    return either a normal matplotlib cmap or a custom cmap
    """
    if name == "YlGnBu_white":
        return LinearSegmentedColormap.from_list(
            "YlGnBu_white",
            [
                "#b97800",
                "#d08b00",
                "#d99826",
                "#e0ab3c",
                "#e9bd52",
                "#f1cf6d",
                "#f6dd8a",
                "#ecef98",
                "#f3f7b2",
                "#f9fdcc",
                # white centre
                "#ffffff",

                "#b4e2b6",
                "#85cfba",
                "#5ac0c0",
                "#39aec3",
                "#2196c1",
                "#2076b3",
                "#2356a4",
                "#233c98",
                "#172978",
                "#081c59",
            ],
            N=256,
        )
    elif name == "ygb_custom":
        return LinearSegmentedColormap.from_list(
            "ygb_custom",
            [
                "#ffffff",
                "#f9fdcc",
                "#edf8b3",
                "#d6efb3",
                "#b4e2b6",
                "#85cfba",
                "#5cc0c0",
                "#39aec3",
                "#2196c1",
                "#2076b3",
                "#2356a4",
                "#233c99",
                "#172978",
                "#071d58",
            ],
            N=256,
        )
    elif name == "Spectral_wWhite":
        return LinearSegmentedColormap.from_list(
            "Spectral_wWhite",
            [
                "#5f4fa2",
                "#3b7db7",
                "#4199b6",
                "#5ab4aa",
                "#88d0a3",
                "#b8e1a0",
                "#e1f399",
                "#ffffff", # white centre
                "#fbdc89",
                "#f9b96a",
                "#f78b52",
                "#ee6246",
                "#d7414d",
                "#b11748",
                "#9e0342",

            ],
            N=256,
        )
    else:
        return mpl.cm.get_cmap(name)