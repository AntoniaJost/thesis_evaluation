# The following script has been written by ChatGPT to provide a fast overview
# of the min and max values of all datasets. It uses functions and code that
# already exist in the evaluation package, but is adapted for this CSV summary
# use case.

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from hydra import compose, initialize_config_dir

# allow imports like: from evaluation.general_functions import ...
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from general_functions import open_model_da, open_era5_da  # noqa: E402


# ============================================================
# USER SETTINGS
# ============================================================
DEFAULT_SUFFIXES = [
    "_ALL",
    "_FRc+FR15",
    "_sst0+AW",
    "_sst0+AW+FRc+FR15",
    "_sst0+AW+sst2+sst4",
    "_sst2+sst4",
]

SUFFIX_TO_MODELS = {
    "_ALL": [
        "free_run_control",
        "free_run_prediction",
        "forced_sst",
        "forced_sst_2k",
        "forced_sst_4k",
        "archesweather",
    ],
    "_FRc+FR15": ["free_run_control", "free_run_prediction"],
    "_sst0+AW": ["forced_sst", "archesweather"],
    "_sst0+AW+FRc+FR15": ["forced_sst", "archesweather", "free_run_control", "free_run_prediction"],
    "_sst0+AW+sst2+sst4": ["forced_sst", "archesweather", "forced_sst_2k", "forced_sst_4k"],
    "_sst2+sst4": ["forced_sst_2k", "forced_sst_4k"],
}

# keep in line with your plotting setup / range_summary defaults
FREQ = "monthly"   # "monthly" or "daily"
GRID = "gn"
START = "1979-01-01"
END = "2024-12-31"

PLEVS = [
    "surface",
    5000, 10000, 15000, 20000, 25000, 30000,
    40000, 50000, 60000, 70000, 85000, 92500, 100000,
]

OUTPUT_TEMPLATE_ABS = "outputs/range_wind/wind_speed_bounds{suffix}.csv"
OUTPUT_TEMPLATE_DIFF = "outputs/range_wind/wind_speed_diff_bounds{suffix}.csv"

# use float32 to keep the temporary memmap smaller
WORK_DTYPE = np.float32

FIELDNAMES = [
    "var",
    "plev",
    "plev_pa",
    "plev_hpa",
    "raw_min",
    "raw_max",
    "p01",
    "p99",
    "p05",
    "p95",
]


# ============================================================
# CONFIG / HELPERS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute wind-speed plotting bounds from actual uas/vas and ua/va data. "
            "Writes CSV rows progressively so partial progress is preserved."
        )
    )
    parser.add_argument(
        "--suffix",
        nargs="+",
        default=DEFAULT_SUFFIXES,
        help=(
            "One or more suffix tags to process. "
            f"Defaults to: {' '.join(DEFAULT_SUFFIXES)}"
        ),
    )
    return parser.parse_args()


def load_cfg(repo_root: Path):
    config_dir = repo_root / "evaluation" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config")
    return cfg


def wind_var_names(plev):
    if plev == "surface":
        return "uas", "vas", None, "surface"
    plev_pa = int(plev)
    return "ua", "va", plev_pa, f"{int(plev_pa / 100)}hPa"


def normalise_monthly_time(da):
    """
    Replace each timestamp by the first day of its month at 00:00,
    so monthly data with different timestamp conventions can align.
    """
    if "time" not in da.coords:
        return da

    t = pd.to_datetime(da["time"].values)
    t_month = t.to_period("M").to_timestamp(how="start")
    return da.assign_coords(time=t_month)


def iter_model_speed_arrays(cfg, model_names: list[str], plev, *, difference: bool):
    """
    Yield one wind-speed DataArray per model member.
    For difference=True, yields (model wind speed - ERA5 wind speed),
    after monthly time normalisation and coordinate alignment.
    """
    u_var, v_var, plev_arg, _ = wind_var_names(plev)

    era5_speed = None
    if difference:
        era5_u = open_era5_da(cfg, var=u_var, start=START, end=END, plev=plev_arg, freq="monthly", grid="gn")
        era5_v = open_era5_da(cfg, var=v_var, start=START, end=END, plev=plev_arg, freq="monthly", grid="gn")

        if FREQ == "monthly":
            era5_u = normalise_monthly_time(era5_u)
            era5_v = normalise_monthly_time(era5_v)

        era5_speed = np.sqrt(era5_u**2 + era5_v**2).astype("float32")

        del era5_u
        del era5_v

    for model_name in model_names:
        model_cfg = cfg.datasets.models[model_name]

        for member in cfg.members:
            u = open_model_da(
                model_cfg=model_cfg,
                cfg=cfg,
                member=member,
                var=u_var,
                modelname=model_cfg.modelname,
                freq=FREQ,
                start=START,
                end=END,
                grid=GRID,
                plev=plev_arg,
            )
            v = open_model_da(
                model_cfg=model_cfg,
                cfg=cfg,
                member=member,
                var=v_var,
                modelname=model_cfg.modelname,
                freq=FREQ,
                start=START,
                end=END,
                grid=GRID,
                plev=plev_arg,
            )

            if FREQ == "monthly":
                u = normalise_monthly_time(u)
                v = normalise_monthly_time(v)

            speed = np.sqrt(u**2 + v**2).astype("float32")

            del u
            del v

            if difference:
                speed, era5_for_diff = xr.align(speed, era5_speed, join="inner")

                if speed.sizes.get("time", 0) == 0:
                    raise ValueError(
                        f"No overlapping time after alignment for wind speed, "
                        f"model='{model_name}', member='{member}', plev='{plev}'."
                    )

                speed = (speed - era5_for_diff).astype("float32")
                del era5_for_diff

            yield speed


def total_nvalues(arrays) -> int:
    total = 0
    for da in arrays:
        total += int(da.size)
    return total


def compute_stats_exact(arrays) -> dict[str, float]:
    """
    Compute exact raw / 1-99 / 5-95 statistics from actual wind-speed values
    across all arrays combined, using a temporary on-disk memmap.
    """
    arrays = list(arrays)
    n_total = total_nvalues(arrays)
    if n_total == 0:
        raise ValueError("No values found for this wind-speed calculation.")

    with tempfile.TemporaryDirectory() as tmpdir:
        mm_path = os.path.join(tmpdir, "wind_values.dat")
        mm = np.memmap(mm_path, dtype=WORK_DTYPE, mode="w+", shape=(n_total,))

        pos = 0
        for da in arrays:
            vals = np.asarray(da.values, dtype=WORK_DTYPE).ravel()
            n = vals.size
            mm[pos:pos + n] = vals
            pos += n

        finite = np.isfinite(mm)
        if not np.any(finite):
            raise ValueError("All computed wind-speed values are NaN or non-finite.")
        vals = np.asarray(mm[finite], dtype=np.float64)

        return {
            "raw_min": float(np.nanmin(vals)),
            "raw_max": float(np.nanmax(vals)),
            "p01": float(np.nanpercentile(vals, 1)),
            "p99": float(np.nanpercentile(vals, 99)),
            "p05": float(np.nanpercentile(vals, 5)),
            "p95": float(np.nanpercentile(vals, 95)),
        }


def process_suffix_to_csv(cfg, suffix: str, outfile: Path, *, difference: bool):
    """
    Compute one CSV progressively for one suffix and one mode
    (absolute or model-minus-ERA5 difference).

    Rows are written immediately after each pressure level finishes, so
    partial results survive crashes or job interruptions.
    """
    if suffix not in SUFFIX_TO_MODELS:
        raise ValueError(
            f"Unknown suffix {suffix!r}. Please add it to SUFFIX_TO_MODELS in this script."
        )

    model_names = SUFFIX_TO_MODELS[suffix]
    rows_for_total: list[dict] = []

    outfile.parent.mkdir(parents=True, exist_ok=True)

    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        f.flush()

        for plev in PLEVS:
            _, _, plev_pa, plev_label = wind_var_names(plev)
            print(f"  Computing {'difference' if difference else 'absolute'} stats for {suffix} | {plev_label}")

            stats = compute_stats_exact(
                iter_model_speed_arrays(cfg, model_names, plev, difference=difference)
            )

            row = {
                "var": "wind_speed",
                "plev": plev_label,
                "plev_pa": plev_pa,
                "plev_hpa": (None if plev_pa is None else int(plev_pa / 100)),
                **stats,
            }

            writer.writerow(row)
            f.flush()
            rows_for_total.append(row)

        df = pd.DataFrame(rows_for_total)
        total_row = {
            "var": "wind_speed",
            "plev": "total",
            "plev_pa": None,
            "plev_hpa": None,
            "raw_min": float(df["raw_min"].min()),
            "raw_max": float(df["raw_max"].max()),
            "p01": float(df["p01"].min()),
            "p99": float(df["p99"].max()),
            "p05": float(df["p05"].min()),
            "p95": float(df["p95"].max()),
        }

        writer.writerow(total_row)
        f.flush()


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    suffixes = args.suffix

    cfg = load_cfg(REPO_ROOT)

    for suffix in suffixes:
        print(f"\n=== Processing suffix {suffix} ===")

        out_abs = REPO_ROOT / OUTPUT_TEMPLATE_ABS.format(suffix=suffix)
        out_diff = REPO_ROOT / OUTPUT_TEMPLATE_DIFF.format(suffix=suffix)

        process_suffix_to_csv(cfg, suffix, out_abs, difference=False)
        print(f"Saved progressively to: {out_abs}")

        process_suffix_to_csv(cfg, suffix, out_diff, difference=True)
        print(f"Saved progressively to: {out_diff}")


if __name__ == "__main__":
    main()