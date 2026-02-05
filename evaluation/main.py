import hydra
from omegaconf import DictConfig, OmegaConf

from evaluation.metrics import global_mean, anomalies, bias_map, soi, location_timeseries

REGISTRY = {
    "global_mean": global_mean.run,
    "anomalies": anomalies.run,
    "bias_map": bias_map.run,
    "soi_kde": soi.run,          # use cfg.plots.soi.kde.enabled
    "soi": soi.run,
    "location_timeseries": location_timeseries.run,
}

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.get("print_config", False):
        print(OmegaConf.to_yaml(cfg))

    for name in cfg.run_plots:
        if name not in REGISTRY:
            raise KeyError(f"Unknown plot '{name}'. Available: {list(REGISTRY.keys())}")
        REGISTRY[name](cfg)

if __name__ == "__main__":
    main()
