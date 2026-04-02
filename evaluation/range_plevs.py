# The following script has been written by ChatGPT to provide a fast overview
# of the min and max values of all datasets. It uses functions and code that
# already exist in the evaluation package, but is adapted for this CSV summary
# use case. This version has been adjusted to be much less memory intensive:
# - monthly timestamps are normalised before model-minus-ERA5 differences
# - no full flattened arrays are stored across models/members
# - statistics are computed directly from xarray objects
# - arrays are cast to float32 where appropriate
# - temporary objects are deleted aggressively

from __future__ import annotations

import csv
import gc
import os

import hydra
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import DictConfig

from evaluation.general_functions import (
    conversion_rules,
    open_model_da_raw,
    open_era5_da_raw,
)

# fixed variable list
VARS = ["ta", "ua", "va", "wap", "hus", "zg"]

# pressure levels used by the models
MODEL_PLEVS_PA = [
    5000, 10000, 15000, 20000, 25000, 30000, 40000,
    50000, 60000, 70000, 85000, 92500, 100000,
]


def _select_model_plevs(da, var: str, context: str):
    if "plev" not in da.dims:
        raise ValueError(
            f"Variable '{var}' in {context} has no 'plev' dimension, "
            "but this script expects pressure-level variables only."
        )

    available = np.asarray(da["plev"].values, dtype=float)
    missing = [p for p in MODEL_PLEVS_PA if not np.any(np.isclose(available, p))]

    if missing:
        raise ValueError(
            f"{context}: missing required pressure levels for '{var}'. "
            f"Missing: {missing}. Available: {[float(v) for v in da['plev'].values]}"
        )

    return da.sel(plev=MODEL_PLEVS_PA)


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


def _safe_stats_da(da, prefix: str) -> dict[str, float]:
    """
    Compute statistics directly from an xarray DataArray without flattening the
    entire field into a large NumPy array.
    """
    finite_count = da.count()

    if hasattr(finite_count, "item"):
        finite_count = finite_count.item()

    if finite_count == 0:
        raise ValueError(f"No finite values available for statistics ({prefix}).")

    q = da.quantile([0.01, 0.05, 0.95, 0.99], skipna=True)

    return {
        f"{prefix}_min": float(da.min(skipna=True).item()),
        f"{prefix}_p01": float(q.sel(quantile=0.01).item()),
        f"{prefix}_p05": float(q.sel(quantile=0.05).item()),
        f"{prefix}_p95": float(q.sel(quantile=0.95).item()),
        f"{prefix}_p99": float(q.sel(quantile=0.99).item()),
        f"{prefix}_max": float(da.max(skipna=True).item()),
    }


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    rs_cfg = cfg.range_summary

    models = list(rs_cfg.models_to_process)
    tag = getattr(rs_cfg, "tag", "")

    # fixed settings for this summary script
    start = "1979-01-01"
    end = "2024-12-31"
    freq = "monthly"
    grid = "gn"

    outdir = os.path.join(hydra.utils.get_original_cwd(), cfg.out.dir,"plev_range")
    os.makedirs(outdir, exist_ok=True)

    outfile = os.path.join(outdir, f"range_plevs{tag}.csv")

    fieldnames = [
        "source",
        "model",
        "member",
        "variable",
        "unit",
        "start",
        "end",
        "raw_min",
        "raw_p01",
        "raw_p05",
        "raw_p95",
        "raw_p99",
        "raw_max",
        "diff_to_era5_min",
        "diff_to_era5_p01",
        "diff_to_era5_p05",
        "diff_to_era5_p95",
        "diff_to_era5_p99",
        "diff_to_era5_max",
    ]

    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for var in VARS:
            print(f"\nProcessing variable: {var}")

            unit_for_models = ""

            # envelope summary across all selected model/member rows
            # this is much lighter than storing all full values and concatenating them
            combined_raw_stats_acc = {
                "raw_min": np.inf,
                "raw_p01": np.inf,
                "raw_p05": np.inf,
                "raw_p95": -np.inf,
                "raw_p99": -np.inf,
                "raw_max": -np.inf,
            }
            combined_diff_stats_acc = {
                "diff_to_era5_min": np.inf,
                "diff_to_era5_p01": np.inf,
                "diff_to_era5_p05": np.inf,
                "diff_to_era5_p95": -np.inf,
                "diff_to_era5_p99": -np.inf,
                "diff_to_era5_max": -np.inf,
            }
            any_model_written = False

            # ERA5 once per variable, on model pressure levels and converted
            era5 = open_era5_da_raw(cfg, var, start, end, "monthly", "gn")
            era5 = _select_model_plevs(era5, var=var, context="ERA5")
            era5, era5_unit = conversion_rules(var, era5, cfg, "era5_natural", "")
            era5 = normalise_monthly_time(era5)
            era5 = era5.astype("float32")

            era5_stats = _safe_stats_da(era5, prefix="raw")

            # for ERA5 itself, diff-to-ERA5 is exactly zero everywhere
            writer.writerow(
                {
                    "source": "era5",
                    "model": "ERA5",
                    "member": "",
                    "variable": var,
                    "unit": era5_unit,
                    "start": start,
                    "end": end,
                    "raw_min": era5_stats["raw_min"],
                    "raw_p01": era5_stats["raw_p01"],
                    "raw_p05": era5_stats["raw_p05"],
                    "raw_p95": era5_stats["raw_p95"],
                    "raw_p99": era5_stats["raw_p99"],
                    "raw_max": era5_stats["raw_max"],
                    "diff_to_era5_min": 0.0,
                    "diff_to_era5_p01": 0.0,
                    "diff_to_era5_p05": 0.0,
                    "diff_to_era5_p95": 0.0,
                    "diff_to_era5_p99": 0.0,
                    "diff_to_era5_max": 0.0,
                }
            )
            f.flush()

            for model_name in models:
                model_cfg = cfg.datasets.models[model_name]

                for member in cfg.members:
                    print(f"  {model_name} - {member}")

                    da = open_model_da_raw(
                        model_cfg,
                        cfg,
                        member,
                        var,
                        model_cfg.modelname,
                        freq,
                        start,
                        end,
                        grid=grid,
                    )

                    da = _select_model_plevs(
                        da,
                        var=var,
                        context=f"{model_name}/{member}",
                    )

                    da, unit_for_models = conversion_rules(var, da, cfg, "model", "")
                    da = normalise_monthly_time(da)
                    da = da.astype("float32")

                    raw_stats = _safe_stats_da(da, prefix="raw")

                    da, era5_for_diff = xr.align(da, era5, join="inner")

                    if da.sizes.get("time", 0) == 0:
                        raise ValueError(
                            f"No overlapping time after alignment for var='{var}', "
                            f"model='{model_name}', member='{member}'."
                        )

                    diff = (da - era5_for_diff).astype("float32")
                    diff_stats = _safe_stats_da(diff, prefix="diff_to_era5")

                    writer.writerow(
                        {
                            "source": "model_member",
                            "model": model_name,
                            "member": member,
                            "variable": var,
                            "unit": unit_for_models,
                            "start": start,
                            "end": end,
                            "raw_min": raw_stats["raw_min"],
                            "raw_p01": raw_stats["raw_p01"],
                            "raw_p05": raw_stats["raw_p05"],
                            "raw_p95": raw_stats["raw_p95"],
                            "raw_p99": raw_stats["raw_p99"],
                            "raw_max": raw_stats["raw_max"],
                            "diff_to_era5_min": diff_stats["diff_to_era5_min"],
                            "diff_to_era5_p01": diff_stats["diff_to_era5_p01"],
                            "diff_to_era5_p05": diff_stats["diff_to_era5_p05"],
                            "diff_to_era5_p95": diff_stats["diff_to_era5_p95"],
                            "diff_to_era5_p99": diff_stats["diff_to_era5_p99"],
                            "diff_to_era5_max": diff_stats["diff_to_era5_max"],
                        }
                    )
                    f.flush()

                    combined_raw_stats_acc["raw_min"] = min(combined_raw_stats_acc["raw_min"], raw_stats["raw_min"])
                    combined_raw_stats_acc["raw_max"] = max(combined_raw_stats_acc["raw_max"], raw_stats["raw_max"])

                    combined_raw_stats_acc["raw_p01"] = min(combined_raw_stats_acc["raw_p01"], raw_stats["raw_p01"])
                    combined_raw_stats_acc["raw_p99"] = max(combined_raw_stats_acc["raw_p99"], raw_stats["raw_p99"])

                    combined_raw_stats_acc["raw_p05"] = min(combined_raw_stats_acc["raw_p05"], raw_stats["raw_p05"])
                    combined_raw_stats_acc["raw_p95"] = max(combined_raw_stats_acc["raw_p95"], raw_stats["raw_p95"])

                    combined_diff_stats_acc["diff_to_era5_min"] = min(
                        combined_diff_stats_acc["diff_to_era5_min"], diff_stats["diff_to_era5_min"]
                    )
                    combined_diff_stats_acc["diff_to_era5_max"] = max(
                        combined_diff_stats_acc["diff_to_era5_max"], diff_stats["diff_to_era5_max"]
                    )

                    combined_diff_stats_acc["diff_to_era5_p01"] = min(
                        combined_diff_stats_acc["diff_to_era5_p01"], diff_stats["diff_to_era5_p01"]
                    )
                    combined_diff_stats_acc["diff_to_era5_p99"] = max(
                        combined_diff_stats_acc["diff_to_era5_p99"], diff_stats["diff_to_era5_p99"]
                    )

                    combined_diff_stats_acc["diff_to_era5_p05"] = min(
                        combined_diff_stats_acc["diff_to_era5_p05"], diff_stats["diff_to_era5_p05"]
                    )
                    combined_diff_stats_acc["diff_to_era5_p95"] = max(
                        combined_diff_stats_acc["diff_to_era5_p95"], diff_stats["diff_to_era5_p95"]
                    )

                    any_model_written = True

                    del da
                    del era5_for_diff
                    del diff
                    gc.collect()

            if not any_model_written:
                raise ValueError(f"No model values collected for variable '{var}'.")

            writer.writerow(
                {
                    "source": "models_all_members_combined",
                    "model": "ALL_SELECTED_MODELS",
                    "member": "ALL_SELECTED_MEMBERS",
                    "variable": var,
                    "unit": unit_for_models,
                    "start": start,
                    "end": end,
                    "raw_min": combined_raw_stats_acc["raw_min"],
                    "raw_p01": combined_raw_stats_acc["raw_p01"],
                    "raw_p05": combined_raw_stats_acc["raw_p05"],
                    "raw_p95": combined_raw_stats_acc["raw_p95"],
                    "raw_p99": combined_raw_stats_acc["raw_p99"],
                    "raw_max": combined_raw_stats_acc["raw_max"],
                    "diff_to_era5_min": combined_diff_stats_acc["diff_to_era5_min"],
                    "diff_to_era5_p01": combined_diff_stats_acc["diff_to_era5_p01"],
                    "diff_to_era5_p05": combined_diff_stats_acc["diff_to_era5_p05"],
                    "diff_to_era5_p95": combined_diff_stats_acc["diff_to_era5_p95"],
                    "diff_to_era5_p99": combined_diff_stats_acc["diff_to_era5_p99"],
                    "diff_to_era5_max": combined_diff_stats_acc["diff_to_era5_max"],
                }
            )
            f.flush()

            del era5
            gc.collect()

    print(f"\nSaved plev summary to: {outfile}")


if __name__ == "__main__":
    main()