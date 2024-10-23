
from typing import Any
import pandas as pd
import numpy as np
import json
import os
from tqdm.auto import tqdm
import torch
from src.data.mimic_iii.real_dataset import MIMIC3RealDatasetCollection
from glob import glob
import os
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import joblib
from pytorch_lightning import Trainer
from src.rdd.rdd_rmse import compute_rdd_metrics_for_seed
import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ct_model(
    seed: int, 
    device: torch.device,
    dataset_config: dict,
    model_path: str
):
    args = DictConfig({
        'model': {
            'dim_treatments': '???', 
            'dim_vitals': '???', 
            'dim_static_features': '???', 
            'dim_outcomes': '???', 
            'name': 'CT', 
            'multi': {
                '_target_': 'src.models.ct.CT', 
                'max_seq_length': '65', 
                'seq_hidden_units': 24, 
                'br_size': 22, 
                'fc_hidden_units': 22, 
                'dropout_rate': 0.2, 
                'num_layer': 2, 
                'num_heads': 3, 
                'max_grad_norm': None, 
                'batch_size': 64, 
                'attn_dropout': True, 
                'disable_cross_attention': False, 
                'isolate_subnetwork': '_', 
                'self_positional_encoding': {
                    'absolute': False, 
                    'trainable': True, 
                    'max_relative_position': 30
                }, 
                'optimizer': {
                    'optimizer_cls': 'adam', 
                    'learning_rate': 0.0001, 
                    'weight_decay': 0.0, 
                    'lr_scheduler': False
                }, 
                'augment_with_masked_vitals': True, 
                'tune_hparams': False, 
                'tune_range': 50, 
                'hparams_grid': None, 
                'resources_per_trial': None
            }
        }, 
        'dataset': {
            'val_batch_size': 512, 
            'treatment_mode': 'multilabel', 
            '_target_': 'src.data.MIMIC3RealDatasetCollection', 
            'seed': '${exp.seed}', 
            'name': 'mimic3_real', 
            'path': 'data/processed/all_hourly_data.h5', 
            'min_seq_length': 30, 
            'max_seq_length': 60, 
            'max_number': 5000, 
            'projection_horizon': 5, 
            'split': {'val': 0.15, 'test': 0.15}, 
            'autoregressive': True, 
            'treatment_list': ['vaso', 'vent'], 
            'outcome_list': ['diastolic blood pressure'], 
            'vital_list': ['heart rate', 'red blood cell count', 'sodium', 'mean blood pressure', 'systemic vascular resistance', 'glucose', 'chloride urine', 'glascow coma scale total', 'hematocrit', 'positive end-expiratory pressure set', 'respiratory rate', 'prothrombin time pt', 'cholesterol', 'hemoglobin', 'creatinine', 'blood urea nitrogen', 'bicarbonate', 'calcium ionized', 'partial pressure of carbon dioxide', 'magnesium', 'anion gap', 'phosphorous', 'venous pvo2', 'platelets', 'calcium urine'], 'static_list': ['gender', 'ethnicity', 'age'], 
            'drop_first': False
        }, 
        'exp': {
            'seed': 10, 
            'gpus': [0], 
            'max_epochs': 1, 
            'logging': False, 
            'mlflow_uri': 'http://127.0.0.1:5000', 
            'unscale_rmse': True, 
            'percentage_rmse': False, 
            'alpha': 0.01, 
            'update_alpha': True, 
            'alpha_rate': 'exp', 
            'balancing': 'domain_confusion', 
            'bce_weight': False, 
            'weights_ema': True, 
            'beta': 0.99
        }
    })

    dataset_collection = MIMIC3RealDatasetCollection(
        dataset_config["path"],
        min_seq_length=dataset_config["min_seq_length"],
        max_seq_length=dataset_config["max_seq_length"],
        seed=seed,
        max_number=dataset_config["max_number"],
        split=dataset_config["split"], 
        projection_horizon=dataset_config["projection_horizon"],
        autoregressive=dataset_config["autoregressive"],
        outcome_list=dataset_config["outcome_list"],
        vitals=dataset_config["vital_list"],
        treatment_list=dataset_config["treatment_list"],
        static_list=dataset_config["static_list"]
    )
    dataset_collection.process_data_multi()
    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]
    multimodel = instantiate(args.model.multi, args, dataset_collection, _recursive_=False)
    multimodel.hparams.exp.weights_ema = False
    multimodel.load_state_dict(torch.load(
        os.path.join(model_path, "checkpoints", os.listdir(os.path.join(model_path, "checkpoints"))[0]), 
        map_location=device
    )["state_dict"])

    multimodel_trainer = Trainer(
        max_epochs=args.exp.max_epochs,
        gradient_clip_val=args.model.multi.max_grad_norm
    )
    multimodel.trainer = multimodel_trainer
    multimodel = multimodel.double()
    multimodel = multimodel.eval()
    multimodel = multimodel.to(device)

    return multimodel


def load_models(
    dataset_config: dict[str, Any], 
    device: torch.device,  
    seeds: list[int], 
    tft_models_prefix_pax: str,
    tft_models_definition: dict | None = None, 
    ct_models_path: str | None = None
) -> dict[int, dict[str, tuple[Any, str] | tuple[Any, None]]]:
    models_dict_per_seed = {}
    if tft_models_definition is not None:
        for model_name_prefix, folder_name_prefix, model_class in tft_models_definition:
            for seed_idx, seed in enumerate(seeds):
                models_dict_per_seed.setdefault(seed, {})[f"{model_name_prefix}_{seed_idx}"] = (
                    model_class,
                    glob(os.path.join(
                        tft_models_prefix_pax, 
                        f"{folder_name_prefix}_{seed_idx}", 
                        "checkpoints/*.ckpt"
                    ))[-1],
                )
    if ct_models_path is not None:
        for seed_index, seed in enumerate(seeds):
            models_dict_per_seed.setdefault(seed, {})[f"CT_{seed_index}"] = (
                lambda _seed, _seed_idx: load_ct_model(
                    device=device, 
                    seed=_seed,
                    dataset_config=dataset_config, 
                    model_path=os.path.join(ct_models_path, str(_seed_idx))
                ),
                None
            )
    return models_dict_per_seed

def format_weighted_metric_values(
    rmse_metric_values: np.ndarray,
    mae_metric_values: np.ndarray,
    metrics_weights: list[float],
):
    rmse_weighted_average = np.average(rmse_metric_values, weights=metrics_weights)
    mae_weighted_average = np.average(mae_metric_values, weights=metrics_weights)
    return {
        "rmse": {
            "values": rmse_metric_values,
            "mean": np.mean(rmse_metric_values),
            "std": np.std(rmse_metric_values),
            "weighted_mean": rmse_weighted_average,
            "weighted_std": np.round(np.sqrt(np.average(
                (np.array(rmse_metric_values) - rmse_weighted_average) ** 2,
                weights=metrics_weights)
            ), 8),
        },
        "mae": {
            "values": mae_metric_values,
            "mean": np.mean(mae_metric_values),
            "std": np.std(mae_metric_values),
            "weighted_mean": mae_weighted_average,
            "weighted_std": np.round(np.sqrt(np.average(
                (np.array(mae_metric_values) - mae_weighted_average) ** 2, 
                weights=metrics_weights
            )), 8),
        }
    }


def compute_metrics_from_values(
    rdd_df_per_time_per_seed: dict[int, dict[int, pd.DataFrame]],
    seeds: list[int],
    rdd_path: str,
    rdd_query_str: str,
    top_percent_outliers_selection: float,
    models_dict_per_seed: dict[int, dict[str, tuple[Any, str] | tuple[Any, None]]],
    tft_models_definition: list[tuple[str, str, Any]],
    compute_ct_values: bool,
):
    # initalize result dict
    metrics_dict = {
        "TIME_SHIFTS": list(rdd_df_per_time_per_seed.keys()),
        "RDD_PATH": rdd_path,
        "rdd_dataset_len_per_time_per_seed": {
            time_shift: {
                seed: len(rdd_df_per_time_per_seed[time_shift][seed])
                for seed in seeds
            } for time_shift in rdd_df_per_time_per_seed.keys()
        },
        "rdd_dataset_filtering": {
            "query_str": rdd_query_str,
            "top_percent_outliers_filtereing": top_percent_outliers_selection,
            "total_filtering_percent_per_time_shift": {
                time_shift: {
                    seed: (
                        rdd_df_per_time_per_seed[time_shift][seed].eval(rdd_query_str).mean() 
                        - (top_percent_outliers_selection * 2)
                    )
                    for seed in seeds
                } for time_shift in rdd_df_per_time_per_seed.keys()
            },
        },
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
                "raw_metrics": {
                    "metrics_per_seed": {},
                    "metrics_per_architecture": {},
                },
                "filtered_metrics": {
                    "metrics_per_seed": {},
                    "metrics_per_architecture": {}
                },
            } for time_shift in rdd_df_per_time_per_seed.keys()
        }
        
    }

    # compute metrics
    for time_shift in rdd_df_per_time_per_seed.keys():
        for seed, models_dict in models_dict_per_seed.items():
            seed_df = rdd_df_per_time_per_seed[time_shift][seed]
            for model_name in models_dict.keys():
                # Raw metrics
                metrics_dict["metrics_per_time_shift"][time_shift]["raw_metrics"]["metrics_per_seed"].setdefault(
                    seed, {}
                ).setdefault("rmse", {})[model_name] = (
                    root_mean_squared_error(y_true=seed_df["rdd_delta"], y_pred=seed_df[model_name + "_delta_demand"])
                )
                metrics_dict["metrics_per_time_shift"][time_shift]["raw_metrics"]["metrics_per_seed"][seed].setdefault(
                    "mae", {}
                )[model_name] = mean_absolute_error(
                    y_true=seed_df["rdd_delta"], y_pred=seed_df[model_name + "_delta_demand"]
                )
                if rdd_query_str:
                    # Filtered metrics
                    filtered_df = seed_df.query(rdd_query_str)
                    if top_percent_outliers_selection:
                        q1, q2 = filtered_df["rdd_delta"].quantile(
                            q=[top_percent_outliers_selection, 1 - top_percent_outliers_selection]).values.tolist()
                        filtered_df.query("@q1 <= rdd_delta <= @q2", inplace=True)
                    metrics_dict["metrics_per_time_shift"][time_shift]["filtered_metrics"]["metrics_per_seed"].setdefault(
                        seed, {}
                    ).setdefault("rmse", {})[model_name] = root_mean_squared_error(
                        y_true=filtered_df["rdd_delta"],
                        y_pred=filtered_df[model_name + "_delta_demand"]
                    )
                    metrics_dict["metrics_per_time_shift"][time_shift]["filtered_metrics"]["metrics_per_seed"][seed].setdefault(
                        "mae", {}
                    )[model_name] = mean_absolute_error(
                        y_true=filtered_df["rdd_delta"], y_pred=filtered_df[model_name + "_delta_demand"]
                    )
                    
    
        # format metrics per architecture
        metrics_weights = list(metrics_dict["rdd_dataset_len_per_time_per_seed"][time_shift].values())
        metrics_filtered_weights = [
            metrics_dict["rdd_dataset_filtering"]["total_filtering_percent_per_time_shift"][time_shift][seeds[idx]] * rdd_dataset_len
            for idx, rdd_dataset_len in enumerate(metrics_weights)
        ]

        models_prefix = [model_prefix for model_prefix, *_ in tft_models_definition]
        if compute_ct_values:
            models_prefix.append("CT")

        for model_prefix in models_prefix:
            metrics_dict["metrics_per_time_shift"][time_shift]["raw_metrics"]["metrics_per_architecture"][model_prefix] = format_weighted_metric_values(
                rmse_metric_values=np.round([
                    metrics_dict["metrics_per_time_shift"][time_shift]["raw_metrics"]["metrics_per_seed"][seed]["rmse"][f"{model_prefix}_{idx}"]
                    for idx, seed in enumerate(seeds)
                ], 8).tolist(),
                mae_metric_values=np.round([
                    metrics_dict["metrics_per_time_shift"][time_shift]["raw_metrics"]["metrics_per_seed"][seed]["mae"][f"{model_prefix}_{idx}"]
                    for idx, seed in enumerate(seeds)
                ], 8).tolist(),
                metrics_weights=metrics_weights,
            )
            if rdd_query_str:
                metrics_dict["metrics_per_time_shift"][time_shift]["filtered_metrics"]["metrics_per_architecture"][model_prefix] = format_weighted_metric_values(
                    rmse_metric_values=[
                        metrics_dict["metrics_per_time_shift"][time_shift]["filtered_metrics"]["metrics_per_seed"][seed]["rmse"][f"{model_prefix}_{idx}"]
                        for idx, seed in enumerate(seeds)
                    ],
                    mae_metric_values=[
                        metrics_dict["metrics_per_time_shift"][time_shift]["filtered_metrics"]["metrics_per_seed"][seed]["mae"][f"{model_prefix}_{idx}"]
                        for idx, seed in enumerate(seeds)
                    ],
                    metrics_weights=metrics_filtered_weights,
                )

    # format values as the one displayed in the paper
    metrics_dict["paper_metrics_per_time_shift"] = {
        time_shift: {
            model_name: {
                "average_rmse": np.round(metrics_values["rmse"]["weighted_mean"], 3),
                "std_rmse": np.round(metrics_values["rmse"]["weighted_std"], 3),
                "average_mae": np.round(metrics_values["mae"]["weighted_mean"], 3),
                "std_mae": np.round(metrics_values["mae"]["weighted_std"], 3),
            } 
            for model_name, metrics_values 
            in metrics_dict["metrics_per_time_shift"][time_shift]["filtered_metrics"]["metrics_per_architecture"].items()
        } 
        for time_shift in rdd_df_per_time_per_seed.keys()
    }

    return metrics_dict


@hydra.main(config_name="config.yaml", config_path="./config/", version_base="1.3.2")
def main(args: DictConfig):

    OmegaConf.set_struct(args, False)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    device = torch.device("cuda")
    seeds = list(args.rdd_metrics.seeds)
    n_jobs = len(seeds)
    # fetch checkpoint paths
    models_dict_per_seed = load_models(
        dataset_config=dict(args.dataset),
        tft_models_prefix_pax=args.rdd_metrics.tft_models_prefix_path,
        device=device,
        seeds=seeds,
        tft_models_definition=list(args.rdd_metrics.tft_models),
        ct_models_path=args.rdd_metrics.ct_models_path
    )

    # Reading RDD dataset
    rdd_dataset = pd.read_parquet(args.rdd.destination_file_path).astype({"subject_id": int, "hours_in": int})
    rdd_df_per_time_per_seed: dict[int, dict[int, pd.DataFrame]] = {
        time_shift: {}
        for time_shift in range(args.rdd_metrics.max_time_shift)
    }
    for time_shift in rdd_df_per_time_per_seed.keys():
        results = joblib.Parallel(n_jobs=n_jobs, backend='loky')(
            joblib.delayed(compute_rdd_metrics_for_seed)(
                rdd_dataset_df=rdd_dataset,
                seed=seed,
                seed_idx=seeds.index(seed),
                models_dict=models_dict,
                device=device,
                dataset_config=dict(args.dataset),
                time_shift=time_shift,
            )
            for seed, models_dict in models_dict_per_seed.items()
        )

        # Storing forecasted values
        for seed, seed_rdd_df in results:
            rdd_df_per_time_per_seed[time_shift][seed] = seed_rdd_df
    
    # compute metrics per seed
    metrics_dict = compute_metrics_from_values(
        rdd_df_per_time_per_seed=rdd_df_per_time_per_seed,
        seeds=seeds,
        rdd_path=args.rdd.destination_file_path,
        rdd_query_str=args.rdd_metrics.rdd_query_str,
        top_percent_outliers_selection=args.rdd_metrics.top_percent_outliers_selection,
        models_dict_per_seed=models_dict_per_seed,
        tft_models_definition=list(args.rdd_metrics.tft_models),
        compute_ct_values=bool(args.rdd_metrics.ct_models_path)
    )
    

    # compute metrics per with filtered
    with open(args.rdd_metrics.destination_file_path, "w") as f:
        f.truncate(0)
        f.seek(0)
        json.dump(metrics_dict, f, indent=4)

    logger.info(json.dumps(metrics_dict, indent=4))
                

if __name__ == "__main__":
    main()

