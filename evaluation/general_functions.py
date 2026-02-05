# evaluation/general_functions.py
from __future__ import annotations

import glob
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import xarray as xr


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


def open_model_da(model_cfg, cfg, member: str, var: str, modelname: str, freq: str, start: str, end: str,
                  grid: str = "gn") -> xr.DataArray:
    """
    Opens (xr.open_dataset) single file for given timeframe
    """
    pattern = model_file_pattern(model_cfg, cfg, member=member, var=var, modelname=modelname, freq=freq, grid=grid)
    path = open_single_match(pattern)
    ds = xr.open_dataset(path).sel(time=slice(start, end))
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in {path}. Available: {list(ds.data_vars)}")
    return ds[var]


def conversion_rules(var: str, da: xr.DataArray, cfg, source: str) -> xr.DataArray:
    """
    convert units if entry in config.yaml available
    currently only K <-> °C
    needed for:
      - ta & tas
      - tos from ERA5
    """
    rule = cfg.conversions.get(var, None) if hasattr(cfg, "conversions") else None
    if rule is None:
        return da

    # make sure tos only gets converted for era5
    applies_to = getattr(rule, "applies_to", "both") # default "both"
    # case tos_ERA5: applies_to = "era5" and source = "era5" -> 'era5' != "both" (TRUE) and 'era5' !='era5' (FALSE) -> FALSE (continues)
    # case tos_MODEL: applies_to = "era5" and source = "model" -> 'era5' != "both" (TRUE) and 'era5' !='model' (TRUE) -> TRUE (returns da)
    # case tas_ERA5: applies_to = 'both' and source='model' -> 'both' != 'both' (FALSE) and 'both' !='era5' (TRUE) -> FALSE (continues)
    # case tas_MODEL: applies_to = 'both' and source='era5' -> 'both' != 'both' (FALSE) and 'both' !='model' (TRUE) -> FALSE (continues)
    if applies_to != "both" and applies_to != source:
        return da
    
    op = rule.op
    val = float(rule.value)
    if op == "sub":
        return da - val
    if op == "add":
        return da + val
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
