import hydra
from omegaconf import DictConfig, OmegaConf

from evaluation.general_functions import normalise_list
from evaluation.metrics import global_mean, anomalies, bias_map, diff_map_raw, soi, individual_plots, zonal_mean, wind, seasonal_cycle, yearly_cycle

REGISTRY = {
    "global_mean": global_mean.run,
    "anomalies": anomalies.run,
    "bias_map": bias_map.run,
    "diff_map_raw": diff_map_raw.run,
    "soi": soi.run,
    "individual_plots": individual_plots.run,
    "zonal_mean": zonal_mean.run,
    "wind": wind.run,
    "seasonal_cycle": seasonal_cycle.run,
    "yearly_cycle": yearly_cycle.run
}

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.get("print_config", False):
        print(OmegaConf.to_yaml(cfg))

    for name in normalise_list(cfg.run_plots):
        if name not in REGISTRY:
            raise KeyError(f"Unknown plot '{name}'. Available: {list(REGISTRY.keys())}")
        REGISTRY[name](cfg)

if __name__ == "__main__":
    main()
