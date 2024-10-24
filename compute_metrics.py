
from typing import Any
import numpy as np
from tqdm.auto import tqdm
import torch
from src.data.mimic_iii.real_dataset import MIMIC3RealDatasetCollection
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from src.evaluation.metrics import forecast_tft_values
from torch.utils.data import DataLoader
from src.evaluation.utils import load_evaluation_model, format_models_dict, save_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def format_weighted_metric_values(
    rmse_metric_values: np.ndarray,
    mae_metric_values: np.ndarray,
):
    return {
        "rmse": {
            "values": rmse_metric_values,
            "mean": np.mean(rmse_metric_values),
            "std": np.std(rmse_metric_values),
        },
        "mae": {
            "values": mae_metric_values,
            "mean": np.mean(mae_metric_values),
            "std": np.std(mae_metric_values),
        }
    }


def format_metrics(
    metrics_per_time_per_seed: dict[int, dict[int, dict[str, float]]],
    seeds: list[int],
    models_dict_per_seed: dict[int, dict[str, tuple[Any, str] | tuple[Any, None]]],
    tft_models_definition: list[tuple[str, str, Any]],
    add_ct_values: bool,
):
    # initalize result dict
    metrics_dict = {
        "TIME_SHIFTS": list(metrics_per_time_per_seed.keys()),
        "models_per_seed": {
            seed: {
                model_name: model_path
                for model_name, (_, model_path)
                in models_dict.items() if "CT_" not in model_name
            }
            for seed, models_dict in models_dict_per_seed.items()
        },
        "metrics_per_time_shift": {
            time_shift: {
                "metrics_per_seed": {},
                "metrics_per_architecture": {},
            } for time_shift in metrics_per_time_per_seed.keys()
        }
        
    }

    # compute metrics
    for time_shift in metrics_per_time_per_seed.keys():
        for seed, models_dict in models_dict_per_seed.items():
            metrics = metrics_per_time_per_seed[time_shift][seed]
            for model_name in models_dict.keys():
                model_prefix = "_".join(model_name.split("_")[:-1])
                # Raw metrics
                metrics_dict["metrics_per_time_shift"][time_shift]["metrics_per_seed"].setdefault(
                    seed, {}
                ).setdefault("rmse", {})[model_name] = metrics[model_prefix]["rmse"]
                metrics_dict["metrics_per_time_shift"][time_shift]["metrics_per_seed"][seed].setdefault(
                    "mae", {}
                )[model_name] = metrics[model_prefix]["mae"]                 
    
        # format metrics per architecture
        models_prefix = [model_prefix for model_prefix, *_ in tft_models_definition]
        if add_ct_values:
            models_prefix.append("CT")

        for model_prefix in models_prefix:
            metrics_dict["metrics_per_time_shift"][time_shift]["metrics_per_architecture"][model_prefix] = format_weighted_metric_values(
                rmse_metric_values=np.round([
                    metrics_dict["metrics_per_time_shift"][time_shift]["metrics_per_seed"][seed]["rmse"][f"{model_prefix}_{idx}"]
                    for idx, seed in enumerate(seeds)
                ], 8).tolist(),
                mae_metric_values=np.round([
                    metrics_dict["metrics_per_time_shift"][time_shift]["metrics_per_seed"][seed]["mae"][f"{model_prefix}_{idx}"]
                    for idx, seed in enumerate(seeds)
                ], 8).tolist(),
            )

    # format values as the one displayed in the paper
    metrics_dict["paper_metrics_per_time_shift"] = {
        time_shift: {
            model_name: {
                "average_rmse": np.round(metrics_values["rmse"]["mean"], 3),
                "std_rmse": np.round(metrics_values["rmse"]["std"], 3),
                "average_mae": np.round(metrics_values["mae"]["mean"], 3),
                "std_mae": np.round(metrics_values["mae"]["std"], 3),
            } 
            for model_name, metrics_values 
            in metrics_dict["metrics_per_time_shift"][time_shift]["metrics_per_architecture"].items()
        } 
        for time_shift in metrics_per_time_per_seed.keys()
    }

    return metrics_dict


@hydra.main(config_name="config.yaml", config_path="./config/", version_base="1.3.2")
def main(args: DictConfig):

    OmegaConf.set_struct(args, False)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    device = torch.device("cuda")
    seeds = list(args.metrics.seeds)
    # fetch checkpoint paths
    models_dict_per_seed = format_models_dict(
        dataset_config=dict(args.dataset),
        tft_models_prefix_pax=args.metrics.tft_models_prefix_path,
        device=device,
        seeds=seeds,
        tft_models_definition=list(args.metrics.tft_models),
        ct_models_path=args.metrics.ct_models_path
    )

    metrics_per_time_per_seed: dict[int, dict[int, dict[str, float]]] = {
        time_shift: {
            seed: {}
            for seed in seeds
        }
        for time_shift in range(args.metrics.max_projection_step)
    }


    for seed, models_dict in models_dict_per_seed.items():
        seed_idx = seeds.index(seed)

        dataset_collection = MIMIC3RealDatasetCollection(
            args.dataset.path,
            min_seq_length=args.dataset.min_seq_length,
            max_seq_length=args.dataset.max_seq_length,
            seed=seed,
            max_number=args.dataset.max_number,
            split=args.dataset.split,
            projection_horizon=args.dataset.projection_horizon,
            autoregressive=args.dataset.autoregressive,
            outcome_list=args.dataset.outcome_list,
            vitals=args.dataset.vital_list,
            treatment_list=args.dataset.treatment_list,
            static_list=args.dataset.static_list,
        )
        dataset_collection.process_data_multi()
        dataset = dataset_collection.test_f_multi
        print("DATASET HORIZON ", dataset_collection.projection_horizon)
        test_loader = DataLoader(dataset, batch_size=1024,shuffle=False)
        
        # Forecast and compoute metrics
        for model_name, (model_class, model_path) in models_dict.items():
            model_prefix = "_".join(model_name.split("_")[:-1])
            model = load_evaluation_model(
                model_class=model_class,
                model_name=model_name, 
                seed=seed, 
                seed_idx=seed_idx, 
                time_shift=args.metrics.max_projection_step-1, 
                model_path=model_path, 
                device=device
            )
            y_pred, y_true = forecast_tft_values(model, test_loader, args.dataset.max_seq_length)
            
            losses_rmse = (np.sqrt(np.mean((y_pred-y_true)**2,axis=0))*dataset_collection.test_f_multi.scaling_params['output_stds']).flatten()
            losses_mae = (np.mean(np.abs(y_pred-y_true),axis=0)*dataset_collection.test_f_multi.scaling_params['output_stds']).flatten()
            for time_shift in range(args.metrics.max_projection_step):
                metrics_per_time_per_seed[time_shift][seed][model_prefix] = {
                    "rmse": losses_rmse[time_shift],
                    "mae": losses_mae[time_shift],
                }
    
    # format metrics dict
    metrics_dict = format_metrics(
        metrics_per_time_per_seed=metrics_per_time_per_seed,
        seeds=seeds,
        models_dict_per_seed=models_dict_per_seed,
        tft_models_definition=list(args.metrics.tft_models),
        add_ct_values=bool(args.metrics.ct_models_path)
    )

    # save metrics
    save_metrics(
        destination_file_path=args.metrics.forecast.destination_file_path, 
        metrics_dict=metrics_dict,
        logger=logger
    )
                

if __name__ == "__main__":
    main()

