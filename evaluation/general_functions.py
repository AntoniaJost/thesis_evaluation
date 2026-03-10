# evaluation/general_functions.py
from __future__ import annotations
from omegaconf import ListConfig
import glob
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import xarray as xr
import os


def normalise_list(value):
    """
    convert string or list-like config entry into a list
    """
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]


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


# def open_era5_da_raw(cfg, var: str, start: str, end: str) -> xr.DataArray:
#     # map to ERA5 variable naming
#     if var not in cfg.variables.era5_name:
#         raise KeyError(
#             f"No ERA5 name mapping for var='{var}'. "
#             f"Available mappings: {list(cfg.variables.era5_name.keys())}"
#         )
#     era5_var = cfg.variables.era5_name[var]
#     root = cfg.datasets.era5.root
#     pattern = cfg.datasets.era5.pattern

#     file_var = "ci" if var == "siconc" else var
#     path = f"{root}/{pattern.format(var=file_var)}"

#     ds = xr.open_dataset(path).sel(time=slice(start, end))
#     if era5_var not in ds:
#         raise KeyError(
#             f"ERA5 variable '{era5_var}' not found in {path}. "
#             f"Available: {list(ds.data_vars)}"
#         )
#     return ds[era5_var]


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

def open_era5_da(cfg, var: str, start: str, end: str, plev=None) -> xr.DataArray:
    # map to ERA5 variable naming
    if var not in cfg.variables.era5_name:
        raise KeyError(
            f"No ERA5 name mapping for var='{var}'. "
            f"Available mappings: {list(cfg.variables.era5_name.keys())}"
        )
    era5_var = cfg.variables.era5_name[var]
    root = cfg.datasets.era5.root
    pattern = cfg.datasets.era5.pattern
    file_var = "ci" if var == "siconc" else var # handle case of sea ice conc., which uses other variable from era5 (ci, instead of siconc)
    path = f"{root}/{pattern.format(var=file_var)}"

    ds = xr.open_dataset(path).sel(time=slice(start, end))
    if era5_var not in ds:
        raise KeyError(
            f"ERA5 variable '{era5_var}' not found in {path}. "
            f"Available: {list(ds.data_vars)}"
        )
    da = ds[era5_var]
    da = select_plev_if_needed(da, var=var, plev=plev, context=path)
    return da

def conversion_rules(var: str, da: xr.DataArray, cfg, source: str, unit_default: str = "") -> tuple[xr.DataArray, str]:
    """
    convert units if entry in config.yaml available, updates unit string
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
    if applies_to != "both" and applies_to != source:
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
    raise ValueError(f"Unknown conversion op: {op}")


def ensemble_mean_as_member(member_to_da: Dict[str, xr.DataArray], name: str = "mean") -> Dict[str, xr.DataArray]:
    """
    Compute mean across members and add as an extra "member".
    Assumes all DAs are aligned.
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