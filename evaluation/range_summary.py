# The following script has been written by ChatGPT to provide a fast overview of the min and max values of all datasets. It uses, however, functions and code that have been written by me before and are used in the evaluation.main, but needed some little changes to work for this csv summary idea, that's why they have been rewritten. This is definitely not the nicest and best script, but it works. As it only serves as a helper for getting the plotting ranges for the main calculations, I did not spend a lot of time on this. 

from __future__ import annotations

import os
from typing import Optional, Iterable

import hydra
import numpy as np
import pandas as pd
import xarray as xr

from evaluation.general_functions import (
    model_file_pattern,
    open_single_match,
    conversion_rules,
)

from evaluation.metrics.bias_map import compute_slope_per_gridpoint


def file_exists_skip(path: str, label: str) -> bool:
    """
    Return True if file already exists and computation should be skipped.
    """
    if os.path.exists(path):
        print(f"Skipping existing {label}: {os.path.basename(path)}")
        return True
    return False

# ---------------------------------------------------------------------
# SUMMARY HELPERS
# ---------------------------------------------------------------------

TARGET_PLEVS = [
    5000, 10000, 15000, 20000, 25000, 30000,
    40000, 50000, 60000, 70000, 85000, 92500, 100000
]


def filter_target_plevs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep surface variables (plev_pa is NaN) and selected target pressure levels.
    """
    return df[(df["plev_pa"].isna()) | (df["plev_pa"].isin(TARGET_PLEVS))].copy()


def normalise_monthly_time(da: xr.DataArray) -> xr.DataArray:
    """
    Replace each timestamp by the first day of its month at 00:00,
    so monthly data with different timestamp conventions can align.
    """
    if "time" not in da.coords:
        return da

    t = pd.to_datetime(da["time"].values)
    t_month = t.to_period("M").to_timestamp(how="start")
    return da.assign_coords(time=t_month)


def compute_compact_summary(df_subset: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate one row per (var, plev_pa), combining min/max and percentile envelopes.
    Works for:
      - full mixed dataset
      - model-only subset
      - era5-only subset
    """
    rows = []

    for (var, plev), g in df_subset.groupby(["var", "plev_pa"], dropna=False):

        row = {
            "var": var,
            "plev_pa": plev,
            "plev_hpa": None if pd.isna(plev) else int(plev / 100),
        }

        # global mean
        row["global_mean_min"] = g["global_mean"].min()
        row["global_mean_mean"] = g["global_mean"].mean()
        row["global_mean_max"] = g["global_mean"].max()

        # raw values
        row["raw_min"] = g["raw_min"].min()
        row["raw_p01"] = g["raw_p01"].min()
        row["raw_p05"] = g["raw_p05"].min()
        row["raw_p95"] = g["raw_p95"].max()
        row["raw_p99"] = g["raw_p99"].max()
        row["raw_max"] = g["raw_max"].max()

        # spatial mean
        row["spatial_min"] = g["spatial_mean_min"].min()
        row["spatial_p01"] = g["spatial_mean_p01"].min()
        row["spatial_p05"] = g["spatial_mean_p05"].min()
        row["spatial_p95"] = g["spatial_mean_p95"].max()
        row["spatial_p99"] = g["spatial_mean_p99"].max()
        row["spatial_max"] = g["spatial_mean_max"].max()

        # temporal mean
        row["temporal_min"] = g["temporal_mean_min"].min()
        row["temporal_p01"] = g["temporal_mean_p01"].min()
        row["temporal_p05"] = g["temporal_mean_p05"].min()
        row["temporal_p95"] = g["temporal_mean_p95"].max()
        row["temporal_p99"] = g["temporal_mean_p99"].max()
        row["temporal_max"] = g["temporal_mean_max"].max()

        # slope
        row["slope_mean_min"] = g["slope_global_mean"].min()
        row["slope_mean_mean"] = g["slope_global_mean"].mean()
        row["slope_mean_max"] = g["slope_global_mean"].max()

        row["slope_min"] = g["slope_map_min"].min()
        row["slope_p01"] = g["slope_map_p01"].min()
        row["slope_p05"] = g["slope_map_p05"].min()
        row["slope_p95"] = g["slope_map_p95"].max()
        row["slope_p99"] = g["slope_map_p99"].max()
        row["slope_max"] = g["slope_map_max"].max()

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["var", "plev_pa"], na_position="first")


def write_model_minus_era5_rows(cfg, path: str):
    """
    Build summary rows from actual model-minus-ERA5 difference fields
    and append them directly to CSV so progress is not lost if the job crashes.
    """
    allowed_vars = list(cfg.variables.allowed)

    for var in allowed_vars:
        meta = cfg.variables.meta.get(var, None)
        long_name = meta.long_name if meta else var
        unit_default = meta.unit if meta else ""

        print(f"\nBuilding model-minus-ERA5 rows for var={var}")

        era5_full = open_full_era5_da(cfg, var)
        era5_plevs = get_all_plevs(era5_full)
        del era5_full

        for model_name, model_cfg in cfg.datasets.models.items():
            if (
                cfg.range_summary.models_to_process is not None
                and model_name not in cfg.range_summary.models_to_process
            ):
                continue

            for member in cfg.members:

                sample = open_full_model_da(
                    model_cfg=model_cfg,
                    cfg=cfg,
                    member=member,
                    var=var,
                    modelname=model_cfg.modelname,
                    freq="monthly",
                    grid="gn",
                )
                model_plevs = get_all_plevs(sample)
                del sample

                common_plevs = sorted(
                    set(era5_plevs).intersection(set(model_plevs)),
                    key=lambda x: (-999 if x is None else x)
                )

                for plev in common_plevs:
                    print(f" diff rows: {model_name} {member} var={var} plev={plev}")

                    da_era5 = open_full_era5_da(cfg, var)
                    da_era5 = select_plev(da_era5, plev)
                    da_era5, unit_here = conversion_rules(var, da_era5, cfg, "era5", unit_default)

                    da_model = open_full_model_da(
                        model_cfg=model_cfg,
                        cfg=cfg,
                        member=member,
                        var=var,
                        modelname=model_cfg.modelname,
                        freq="monthly",
                        grid="gn",
                    )
                    da_model = select_plev(da_model, plev)
                    da_model, _ = conversion_rules(var, da_model, cfg, "model", unit_here)

                    da_model = normalise_monthly_time(da_model)
                    da_era5 = normalise_monthly_time(da_era5)
                    da_model, da_era5 = xr.align(da_model, da_era5, join="inner")

                    if da_model.sizes.get("time", 0) == 0:
                        print(f"  skipping {model_name} {member} {var} plev={plev}: no overlapping time after alignment")
                        continue

                    diff_da = da_model - da_era5

                    row = build_summary_row(
                        diff_da,
                        dataset_type="model_minus_era5",
                        dataset_name=model_name,
                        member=member,
                        var=var,
                        long_name=long_name,
                        unit=unit_here,
                        plev=plev,
                    )

                    append_row_csv(row, path)


def write_compact_summaries(cfg, csv_path: str, outdir: str, tag: str = ""):
    """
    Read the full summary CSV and write:
      1) compact summary across all datasets
      2) model-minus-ERA5 compact summary
    Only computes files that do not already exist.
    """
    compact_all_path = os.path.join(outdir, f"range_summary_compact{tag}.csv")
    model_minus_era5_rows_path = os.path.join(outdir, f"model_minus_era5_rows{tag}.csv")
    model_minus_era5_path = os.path.join(
        outdir,
        f"model_minus_era5_summary_by_var_plev{tag}.csv"
    )

    if file_exists_skip(compact_all_path, "compact summary") and file_exists_skip(
        model_minus_era5_path, "model-minus-ERA5 summary"
    ):
        return

    df = pd.read_csv(csv_path)
    df = filter_target_plevs(df)

    # -------------------------------------------------
    # 1) compact summary across all datasets
    # -------------------------------------------------
    if not os.path.exists(compact_all_path):
        compact_all = compute_compact_summary(df)
        compact_all.to_csv(compact_all_path, index=False)
        print(f"Saved compact summary: {compact_all_path}")
    else:
        print(f"Skipping existing compact summary: {os.path.basename(compact_all_path)}")

    # -------------------------------------------------
    # 2) model-minus-ERA5 compact summary
    # -------------------------------------------------
    if not os.path.exists(model_minus_era5_rows_path):
        write_model_minus_era5_rows(cfg, model_minus_era5_rows_path)

        if os.path.exists(model_minus_era5_rows_path):
            print(f"Saved model-minus-ERA5 full rows: {model_minus_era5_rows_path}")
        else:
            print("No model-minus-ERA5 rows file was created.")
    else:
        print(f"Skipping existing model-minus-ERA5 full rows: {os.path.basename(model_minus_era5_rows_path)}")

    if not os.path.exists(model_minus_era5_path):
        if not os.path.exists(model_minus_era5_rows_path):
            print("Skipping model-minus-ERA5 summary because no diff rows file exists.")
            return
        diff_df = pd.read_csv(model_minus_era5_rows_path)
        diff_df = filter_target_plevs(diff_df)

        model_minus_era5 = compute_compact_summary(diff_df)
        model_minus_era5.to_csv(model_minus_era5_path, index=False)
        print(f"Saved model-minus-ERA5 summary: {model_minus_era5_path}")
    else:
        print(f"Skipping existing model-minus-ERA5 summary: {os.path.basename(model_minus_era5_path)}")

# ---------------------------------------------------------------------
# DATA OPENING
# ---------------------------------------------------------------------

def open_full_model_da(
    model_cfg,
    cfg,
    member: str,
    var: str,
    modelname: str,
    freq: str,
    grid: str = "gn",
) -> xr.DataArray:

    pattern = model_file_pattern(
        model_cfg=model_cfg,
        cfg=cfg,
        member=member,
        var=var,
        modelname=modelname,
        freq=freq,
        grid=grid,
    )

    path = open_single_match(pattern)

    with xr.open_dataset(path) as ds:
        if var not in ds:
            raise KeyError(
                f"Variable '{var}' not found in {path}. "
                f"Available: {list(ds.data_vars)}"
            )
        da = ds[var].load()

    return da


def open_full_era5_da(cfg, var: str) -> xr.DataArray:

    if var not in cfg.variables.era5_name:
        raise KeyError(
            f"No ERA5 name mapping for var='{var}'. "
            f"Available mappings: {list(cfg.variables.era5_name.keys())}"
        )

    era5_var = cfg.variables.era5_name[var]
    root = cfg.datasets.era5.root
    pattern = cfg.datasets.era5.pattern
    file_var = "ci" if var == "siconc" else var

    path = f"{root}/{pattern.format(var=file_var)}"

    with xr.open_dataset(path) as ds:
        if era5_var not in ds:
            raise KeyError(
                f"ERA5 variable '{era5_var}' not found in {path}. "
                f"Available: {list(ds.data_vars)}"
            )
        da = ds[era5_var].load()

    return da


# ---------------------------------------------------------------------
# PRESSURE LEVEL HANDLING
# ---------------------------------------------------------------------

def get_all_plevs(da: xr.DataArray) -> list[Optional[float]]:
    if "plev" not in da.dims:
        return [None]
    return [float(v) for v in da["plev"].values]


def select_plev(da: xr.DataArray, plev: Optional[float]) -> xr.DataArray:

    if "plev" not in da.dims:
        return da

    available = np.asarray(da["plev"].values, dtype=float)

    idx = np.where(np.isclose(available, float(plev)))[0]

    if len(idx) == 0:
        raise ValueError(
            f"Requested plev={plev} not found. Available: {available.tolist()}"
        )

    da = da.isel(plev=int(idx[0]))

    # squeeze only if dimension still exists
    if "plev" in da.dims:
        da = da.squeeze("plev", drop=True)

    return da


# ---------------------------------------------------------------------
# STATISTICS
# ---------------------------------------------------------------------

def area_weighted_spatial_mean(da: xr.DataArray):
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights).mean(dim=("lat", "lon"))


def temporal_mean_map(da: xr.DataArray):
    return da.mean("time")


def global_mean_scalar(da: xr.DataArray):
    sm = area_weighted_spatial_mean(da)
    return float(sm.mean("time").values)


def area_weighted_mean_map(da: xr.DataArray):
    weights = np.cos(np.deg2rad(da["lat"]))
    return float(da.weighted(weights).mean(("lat", "lon")).values)


def finite_values(da: xr.DataArray):
    vals = np.asarray(da.values).ravel()
    return vals[np.isfinite(vals)]


def summarise_distribution(values: np.ndarray, prefix: str):

    if values.size == 0:
        return {f"{prefix}_{k}": np.nan for k in
                ["min", "max", "p01", "p05", "p95", "p99"]}

    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_p01": float(np.percentile(values, 1)),
        f"{prefix}_p05": float(np.percentile(values, 5)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
    }


def slope_map_per_decade(da: xr.DataArray):
    return compute_slope_per_gridpoint(da) * 10.0


# ---------------------------------------------------------------------
# CSV HELPERS
# ---------------------------------------------------------------------

def plev_to_hpa(plev):
    if plev is None:
        return None
    return float(plev) / 100.0


def append_row_csv(row: dict, path: str):
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


# ---------------------------------------------------------------------
# SUMMARY ROW
# ---------------------------------------------------------------------

def build_summary_row(
    da,
    dataset_type,
    dataset_name,
    member,
    var,
    long_name,
    unit,
    plev,
):

    spatial_vals = finite_values(area_weighted_spatial_mean(da))
    temporal_vals = finite_values(temporal_mean_map(da))
    raw_vals = finite_values(da)

    slope_map = slope_map_per_decade(da)
    slope_vals = finite_values(slope_map)

    row = dict(
        dataset_type=dataset_type,
        dataset_name=dataset_name,
        member=member,
        var=var,
        long_name=long_name,
        unit=unit,
        slope_unit=f"{unit}/decade",
        plev_pa=plev,
        plev_hpa=plev_to_hpa(plev),
        global_mean=global_mean_scalar(da),
        slope_global_mean=area_weighted_mean_map(slope_map),
    )

    row.update(summarise_distribution(raw_vals, "raw"))
    row.update(summarise_distribution(spatial_vals, "spatial_mean"))
    row.update(summarise_distribution(temporal_vals, "temporal_mean"))
    row.update(summarise_distribution(slope_vals, "slope_map"))

    return row


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):

    TAG = str(cfg.range_summary.tag)
    outdir = os.path.join(
        hydra.utils.get_original_cwd(),
        cfg.out.dir,
        "range_summary",
    )
    os.makedirs(outdir, exist_ok=True)

    csv_path = os.path.join(outdir, f"range_summary{TAG}.csv")
    full_exists = file_exists_skip(csv_path, "full summary")

    if not full_exists:
        allowed_vars = list(cfg.variables.allowed)

        for var in allowed_vars:

            meta = cfg.variables.meta.get(var, None)
            long_name = meta.long_name if meta else var
            unit_default = meta.unit if meta else ""

            print(f"\nProcessing variable: {var}")

            # -------------------------------------------------
            # ERA5
            # -------------------------------------------------

            print(" Discovering ERA5 pressure levels")

            era5_full = open_full_era5_da(cfg, var)
            era5_plevs = get_all_plevs(era5_full)
            del era5_full

            for plev in era5_plevs:

                print(f" ERA5 plev={plev}")

                da = open_full_era5_da(cfg, var)
                da = select_plev(da, plev)
                da, unit_here = conversion_rules(var, da, cfg, "era5", unit_default)

                row = build_summary_row(
                    da,
                    "era5",
                    "era5",
                    None,
                    var,
                    long_name,
                    unit_here,
                    plev,
                )

                append_row_csv(row, csv_path)

            # -------------------------------------------------
            # MODELS
            # -------------------------------------------------

            for model_name, model_cfg in cfg.datasets.models.items():
                if cfg.range_summary.models_to_process is not None and model_name not in cfg.range_summary.models_to_process:
                    continue
                for member in cfg.members:

                    print(f" Discovering plevs for {model_name} {member}")

                    sample = open_full_model_da(
                        model_cfg=model_cfg,
                        cfg=cfg,
                        member=member,
                        var=var,
                        modelname=model_cfg.modelname,
                        freq="monthly",
                        grid="gn",
                    )

                    plevs = get_all_plevs(sample)
                    del sample

                    for plev in plevs:

                        print(f" {model_name} {member} plev={plev}")

                        da = open_full_model_da(
                            model_cfg=model_cfg,
                            cfg=cfg,
                            member=member,
                            var=var,
                            modelname=model_cfg.modelname,
                            freq="monthly",
                            grid="gn",
                        )

                        da = select_plev(da, plev)

                        da, unit_here = conversion_rules(
                            var, da, cfg, "model", unit_default
                        )

                        row = build_summary_row(
                            da,
                            "model",
                            model_name,
                            member,
                            var,
                            long_name,
                            unit_here,
                            plev,
                        )

                        append_row_csv(row, csv_path)

    print("\nWriting compact summary files...")
    write_compact_summaries(cfg=cfg, csv_path=csv_path, outdir=outdir, tag=TAG)
    print("All files are there/done.")

if __name__ == "__main__":
    main()
