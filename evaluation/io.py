# evaluation/io.py
from __future__ import annotations

import glob
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import xarray as xr


def open_single_match(pattern: str) -> str:
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No files matched pattern:\n{pattern}")
    if len(matches) > 1:
        # keep deterministic: pick lexicographically smallest
        matches = sorted(matches)
    return matches[0]


def resolve_period(cfg, plot_cfg) -> Tuple[str, str]:
    """
    Priority:
      1) plot_cfg.time.use_named -> cfg.periods.named[...]
      2) plot_cfg.time.start/end (both not None)
      3) cfg.periods.default
    """
    use_named = getattr(plot_cfg.time, "use_named", None)
    if use_named:
        p = cfg.periods.named[use_named]
        return p.start, p.end

    start = getattr(plot_cfg.time, "start", None)
    end = getattr(plot_cfg.time, "end", None)
    if start and end:
        return start, end

    return cfg.periods.default.start, cfg.periods.default.end


def model_file_pattern(model_cfg, member: str, var: str, freq: str, grid: str = "gn") -> str:
    """
    Expected structure (your CMORised AWM):
      {root}/{member}/{Amon|day}/{var}/{grid}/{filename}
    """
    if freq not in ("monthly", "daily"):
        raise ValueError(f"freq must be 'monthly' or 'daily', got: {freq}")

    table = model_cfg.freq_map[freq]  # Amon or day
    if freq == "monthly":
        pat = model_cfg.pattern_monthly.format(var=var)
    else:
        pat = model_cfg.pattern_daily.format(var=var)

    return f"{model_cfg.root}/{member}/{table}/{var}/{grid}/{pat}"


def open_model_da(model_cfg, member: str, var: str, freq: str, start: str, end: str,
                  grid: str = "gn", var_in_file: Optional[str] = None) -> xr.DataArray:
    pattern = model_file_pattern(model_cfg, member=member, var=var, freq=freq, grid=grid)
    path = open_single_match(pattern)
    ds = xr.open_dataset(path).sel(time=slice(start, end))
    v = var_in_file or var
    if v not in ds:
        raise KeyError(f"Variable '{v}' not found in {path}. Available: {list(ds.data_vars)}")
    return ds[v]


def open_era5_da(cfg, var: str, start: str, end: str) -> xr.DataArray:
    """
    ERA5 file structure:
      {cfg.datasets.era5.root}/{cfg.datasets.era5.pattern}
    and variable name mapping in cfg.variables.era5_name
    """
    era5_var = cfg.variables.era5_name[var]
    path = f"{cfg.datasets.era5.root}/{cfg.datasets.era5.pattern.format(var=var)}"
    path = open_single_match(path)
    ds = xr.open_dataset(path).sel(time=slice(start, end))
    if era5_var not in ds:
        raise KeyError(f"ERA5 variable '{era5_var}' not found in {path}. Available: {list(ds.data_vars)}")
    return ds[era5_var]


def maybe_convert_units(var: str, da: xr.DataArray, cfg) -> xr.DataArray:
    """
    Keep your existing rules but centralised:
      - ta, tas: K -> °C
      - tos: depending on your ERA5 preparation; you previously used K->°C for ERA5 tos.
    """
    rule = cfg.conversions.get(var, None) if hasattr(cfg, "conversions") else None
    if rule is None:
        return da

    # supported: {"op": "add", "value": ...} or {"op": "sub", ...}
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
