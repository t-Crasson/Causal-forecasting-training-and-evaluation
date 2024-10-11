

from typing import Any
import pandas as pd
import numpy as np
import json
import yaml
import os
from tqdm.auto import tqdm
import torch
from src.data.mimic_iii.real_dataset import MIMIC3RealDatasetCollection, MIMIC3RealDataset
from src.data.rdd import _indexes_day_x_both_ways_iterator
from torch.utils.data import DataLoader
from src.models.baseline import baseline
from src.models.model_m_e_theta import m_e_theta_daily
from glob import glob
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings("ignore")
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning import Trainer

SEEDS = [10,101,1001,10010,10110]
SEEDS = [10, 101]
N_JOBS = len(SEEDS)


# RDD_PATH = "/home/thomas/mimic/physionet.org/files/mimiciii/causal-forecasting-bis/rdd/data/mimic_ct_formatted_rdd_values_all_seeds.parquet"
# RDD_PATH = "/home/thomas/mimic/physionet.org/files/mimiciii/causal-forecasting-bis/rdd/data/rdd_multi_method.parquet"
# RDD_PATH = "/home/thomas/mimic/physionet.org/files/mimiciii/causal-forecasting-bis/rdd/data/rdd_raw_test.parquet"
RDD_PATH = "/home/thomas/mimic/physionet.org/files/mimiciii/causal-forecasting-bis/rdd/data/rdd_raw_test_with_cv.parquet"
# RDD_PATH = "/home/thomas/mimic/physionet.org/files/mimiciii/causal-forecasting-bis/rdd/data/rdd_raw_test_cv.parquet"
TIME_SHIFT = 0

# TFT_MODELS_PREFIX_PATH = "/home/thomas/mimic/physionet.org/files/mimiciii/CausalTransformer"
TFT_MODELS_PREFIX_PATH = "/home/thomas/fork_causal_transformer/Causal-forecasting-training-and-evaluation"

COMPUTE_CT_VALUES = True

TARGET_PATH = "/home/thomas/fork_causal_transformer/Causal-forecasting-training-and-evaluation/rdd_metrics"
RDD_METHODS = [
    "single_linear_ridge_model_linear_limit_5",
    "single_linear_ridge_model_linear_limit_10",
    "single_linear_ridge_model_linear_weight_5",
    "single_linear_ridge_model_linear_weight_10",
    "cv_linear_weight_20",
]
RDD_QUERY_STR = "left_days >= 3 and right_days >= 3"
TOP_PERCENT_OUTLIERS_SECLECTION = .025
# METRICS_FILE_SUFFIX = "test_bug_density_pipeline"
METRICS_FILE_SUFFIX = "cdf"
# METRICS_FILE_SUFFIX = "test_thomas_decay"
TFT_MODELS = [
    # (
    #     "TFT_density_repro_pipeline", 
    #     "TFT_deterministic/m_e_density_repro_pipeline", 
    #     #"TFT/m_e_density_large_final_correc", 
    #     m_e_theta_daily,
    #     [(SEEDS[idx], idx) for idx in range(len(SEEDS))]
    # ),
    # (
    #     "TFT_density_repro", 
    #     "TFT_deterministic/theta_density_repro", 
    #     #"TFT/m_e_density_large_final_correc", 
    #     m_e_theta_daily,
    #     [(SEEDS[idx], idx) for idx in range(len(SEEDS))]
    # ),
    (
        "TFT_density_paper", 
        "TFT_deterministic/theta_density_decay_-5", 
        #"TFT/m_e_density_large_final_correc", 
        m_e_theta_daily,
        [(SEEDS[idx], idx) for idx in range(len(SEEDS))]
    ),
    (
        "TFT_density_lower", 
        "TFT_deterministic/theta_density_low_lr", 
        #"TFT/m_e_density_large_final_correc", 
        m_e_theta_daily,
        [(SEEDS[idx], idx) for idx in range(len(SEEDS))]
    ),
    # (
    #     "TFT_cdf", 
    #     "TFT_deterministic/theta_cdf_low_lr_bis", 
    #     #"TFT/m_e_density_large_final_correc", 
    #     m_e_theta_daily,
    #     [(SEEDS[idx], idx) for idx in range(len(SEEDS))]
    # ),
    # (
    #     "TFT_cdf_low_lr", 
    #     "TFT_deterministic/theta_cdf_low_lr_bis_1", 
    #     #"TFT/m_e_density_large_final_correc", 
    #     m_e_theta_daily,
    #     [(SEEDS[idx], idx) for idx in range(len(SEEDS))]
    # ),
    (
        "TFT_cdf_big_lr", 
        "TFT_deterministic/theta_cdf_bigger_lr", 
        #"TFT/m_e_density_large_final_correc", 
        m_e_theta_daily,
        [(SEEDS[idx], idx) for idx in range(len(SEEDS))]
    ),
    (
        "TFT_cdf_paper", 
        "TFT_repro_compare/m_e_cdf_night", 
        # "TFT_drop/theta_cdf_no_decay", 
        m_e_theta_daily,
        [(SEEDS[idx], idx) for idx in range(len(SEEDS))]
    ),
    # (
    #     # name in final dataframe
    #     "TFT_baseline", 
    #     # folder path
    #     "TFT/baseline_large_multi_repro_bis_2", 
    #     # model class
    #     baseline, [
    #         # (seed, index name in file name)
    #         # change thos values if a model does not exist yet
    #         (SEEDS[idx], idx) for idx in range(len(SEEDS))
    #     ]
    # ),
    
]

def load_ct_model(seed, seed_idx, device):


    print(f"LOADING CT MODEL WITH SEED {seed}")
    with open('/home/thomas/mimic/physionet.org/files/mimiciii/CausalTransformer/config/dataset/mimic3_real.yaml', 'r') as file:
        ct_config = yaml.safe_load(file)["dataset"]
    with open('/home/thomas/mimic/physionet.org/files/mimiciii/CausalTransformer/config/backbone/ct_hparams/mimic3_real/diastolic_blood_pressure.yaml', 'r') as file:
        config_MODEL = yaml.safe_load(file)
    args = DictConfig({'model': {'dim_treatments': '???', 'dim_vitals': '???', 'dim_static_features': '???', 'dim_outcomes': '???', 'name': 'CT', 'multi': {'_target_': 'src.models.ct.CT', 'max_seq_length': '65', 'seq_hidden_units': 24, 'br_size': 22, 'fc_hidden_units': 22, 'dropout_rate': 0.2, 'num_layer': 2, 'num_heads': 3, 'max_grad_norm': None, 'batch_size': 64, 'attn_dropout': True, 'disable_cross_attention': False, 'isolate_subnetwork': '_', 'self_positional_encoding': {'absolute': False, 'trainable': True, 'max_relative_position': 30}, 'optimizer': {'optimizer_cls': 'adam', 'learning_rate': 0.0001, 'weight_decay': 0.0, 'lr_scheduler': False}, 'augment_with_masked_vitals': True, 'tune_hparams': False, 'tune_range': 50, 'hparams_grid': None, 'resources_per_trial': None}}, 'dataset': {'val_batch_size': 512, 'treatment_mode': 'multilabel', '_target_': 'src.data.MIMIC3RealDatasetCollection', 'seed': '${exp.seed}', 'name': 'mimic3_real', 'path': 'data/processed/all_hourly_data.h5', 'min_seq_length': 30, 'max_seq_length': 60, 'max_number': 5000, 'projection_horizon': 5, 'split': {'val': 0.15, 'test': 0.15}, 'autoregressive': True, 'treatment_list': ['vaso', 'vent'], 'outcome_list': ['diastolic blood pressure'], 'vital_list': ['heart rate', 'red blood cell count', 'sodium', 'mean blood pressure', 'systemic vascular resistance', 'glucose', 'chloride urine', 'glascow coma scale total', 'hematocrit', 'positive end-expiratory pressure set', 'respiratory rate', 'prothrombin time pt', 'cholesterol', 'hemoglobin', 'creatinine', 'blood urea nitrogen', 'bicarbonate', 'calcium ionized', 'partial pressure of carbon dioxide', 'magnesium', 'anion gap', 'phosphorous', 'venous pvo2', 'platelets', 'calcium urine'], 'static_list': ['gender', 'ethnicity', 'age'], 'drop_first': False}, 'exp': {'seed': 10, 'gpus': [0], 'max_epochs': 1, 'logging': False, 'mlflow_uri': 'http://127.0.0.1:5000', 'unscale_rmse': True, 'percentage_rmse': False, 'alpha': 0.01, 'update_alpha': True, 'alpha_rate': 'exp', 'balancing': 'domain_confusion', 'bce_weight': False, 'weights_ema': True, 'beta': 0.99}})

    dataset_collection = MIMIC3RealDatasetCollection("data/processed/all_hourly_data.h5",min_seq_length=30,max_seq_length=60,
                                                        seed=seed,max_number=10000,split = {"val":0.15,"test":0.15}, projection_horizon=5,autoregressive=True,
                                                        outcome_list = config["outcome_list"],
                                                        vitals = config["vital_list"],
                                                        treatment_list = config["treatment_list"],
                                                        static_list = config["static_list"]
                                                        )
    dataset_collection.process_data_multi()
    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]
    multimodel = instantiate(args.model.multi, args, dataset_collection, _recursive_=False)
    multimodel.hparams.exp.weights_ema = False
    file = os.listdir(f"/home/thomas/mimic/physionet.org/files/mimiciii/CausalTransformer/multirun/2024-09-10/02-17-13/{seed_idx}/checkpoints/")[0]
    path = os.path.join(f"/home/thomas/mimic/physionet.org/files/mimiciii/CausalTransformer/multirun/2024-09-10/02-17-13/{seed_idx}/checkpoints",file)
    state_dict = torch.load(path, map_location=device)["state_dict"]
    multimodel.load_state_dict(state_dict)

    multimodel_trainer = Trainer(gpus=eval(str(args.exp.gpus)), max_epochs=args.exp.max_epochs,
                                    #terminate_on_nan=True,
                                    gradient_clip_val=args.model.multi.max_grad_norm)
    multimodel.trainer = multimodel_trainer
    multimodel = multimodel.double()
    multimodel = multimodel.eval()
    multimodel = multimodel.to(device)

    multimodel.hparams.dataset.projection_horizon = 1 + TIME_SHIFT  

    return multimodel

def process_data_rdd(mimic_dataset: MIMIC3RealDataset, time_shift: int = 0):
    assert mimic_dataset.processed
    assert time_shift >= 0

    dict_mapping = mimic_dataset.rdd_dict_mapping

    outputs = mimic_dataset.data['outputs']
    prev_outputs = mimic_dataset.data['prev_outputs']
    sequence_lengths = mimic_dataset.data['sequence_lengths']
    vitals = mimic_dataset.data['vitals']
    next_vitals = mimic_dataset.data['next_vitals']
    active_entries = mimic_dataset.data['active_entries']
    current_treatments = mimic_dataset.data['current_treatments']
    previous_treatments = mimic_dataset.data['prev_treatments']
    static_features = mimic_dataset.data['static_features']
    if 'stabilized_weights' in mimic_dataset.data:
        stabilized_weights = mimic_dataset.data['stabilized_weights']
    subject_ids = mimic_dataset.data['subject_ids']

    num_patients, max_seq_length, num_features = outputs.shape
    num_seq2seq_rows = num_patients * max_seq_length

    seq2seq_previous_treatments = np.zeros((num_seq2seq_rows, max_seq_length, previous_treatments.shape[-1]))
    seq2seq_current_treatments = np.zeros((num_seq2seq_rows, max_seq_length, current_treatments.shape[-1]))
    seq2seq_static_features = np.zeros((num_seq2seq_rows, static_features.shape[-1]))
    seq2seq_outputs = np.zeros((num_seq2seq_rows, max_seq_length, outputs.shape[-1]))
    seq2seq_prev_outputs = np.zeros((num_seq2seq_rows, max_seq_length, prev_outputs.shape[-1]))
    seq2seq_vitals = np.zeros((num_seq2seq_rows, max_seq_length, vitals.shape[-1]))
    seq2seq_next_vitals = np.zeros((num_seq2seq_rows, max_seq_length - 1, next_vitals.shape[-1]))
    seq2seq_active_entries = np.zeros((num_seq2seq_rows, max_seq_length, active_entries.shape[-1]))
    seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)
    seq2seq_subject_ids = np.zeros(num_seq2seq_rows)
    if 'stabilized_weights' in mimic_dataset.data:
        seq2seq_stabilized_weights = np.zeros((num_seq2seq_rows, max_seq_length))

    total_seq2seq_rows = 0  # we use this to shorten any trajectories later

    for i in range(num_patients):
        sequence_length = int(sequence_lengths[i])
        signal_id = subject_ids[i]
        for (
            sequence_index,
            day_x_min_index,
            day_x_max_index,
            next_day_x_min_index,
            next_day_x_max_index,
            prediction_index,
        ) in _indexes_day_x_both_ways_iterator(dict_mapping=dict_mapping, od_id=signal_id):
            
            # t = prediction_index + projection_horizon - 1
            t = prediction_index
            if (t >= sequence_length) or (t - time_shift < 0):
                # TODO je crois qu'on peut zapper ce cas et que normalement ça marche
                continue

            # add reference sequence
            seq2seq_active_entries[total_seq2seq_rows, :(t + 1), :] = active_entries[i, :(t + 1), :]
            if 'stabilized_weights' in mimic_dataset.data:
                seq2seq_stabilized_weights[total_seq2seq_rows, :(t + 1)] = stabilized_weights[i, :(t + 1)]
            seq2seq_previous_treatments[total_seq2seq_rows, :(t + 1), :] = previous_treatments[i, :(t + 1), :]
            seq2seq_current_treatments[total_seq2seq_rows, :(t + 1), :] = current_treatments[i, :(t + 1), :]
            seq2seq_outputs[total_seq2seq_rows, :(t + 1), :] = outputs[i, :(t + 1), :]
            seq2seq_prev_outputs[total_seq2seq_rows, :(t + 1), :] = prev_outputs[i, :(t + 1), :]
            seq2seq_vitals[total_seq2seq_rows, :(t + 1), :] = vitals[i, :(t + 1), :]
            seq2seq_next_vitals[total_seq2seq_rows, :min(t + 1, sequence_length - 1), :] = next_vitals[i, :min(t + 1, sequence_length - 1), :]
            seq2seq_sequence_lengths[total_seq2seq_rows] = t + 1
            seq2seq_static_features[total_seq2seq_rows] = static_features[i]
            seq2seq_subject_ids[total_seq2seq_rows] = signal_id

            total_seq2seq_rows += 1

            # add modified sequence with previous treatment as current treatment
            seq2seq_active_entries[total_seq2seq_rows, :(t + 1), :] = active_entries[i, :(t + 1), :]
            if 'stabilized_weights' in mimic_dataset.data:
                seq2seq_stabilized_weights[total_seq2seq_rows, :(t + 1)] = stabilized_weights[i, :(t + 1)]
            seq2seq_current_treatments[total_seq2seq_rows, :(t + 1), :] = current_treatments[i, :(t + 1), :]
            seq2seq_previous_treatments[total_seq2seq_rows, :(t + 1), :] = previous_treatments[i, :(t + 1), :]
            seq2seq_outputs[total_seq2seq_rows, :(t + 1), :] = outputs[i, :(t + 1), :]
            seq2seq_prev_outputs[total_seq2seq_rows, :(t + 1), :] = prev_outputs[i, :(t + 1), :]
            seq2seq_vitals[total_seq2seq_rows, :(t + 1), :] = vitals[i, :(t + 1), :]
            seq2seq_next_vitals[total_seq2seq_rows, :min(t + 1, sequence_length - 1), :] = next_vitals[i, :min(t + 1, sequence_length - 1), :]
            seq2seq_sequence_lengths[total_seq2seq_rows] = t + 1
            seq2seq_static_features[total_seq2seq_rows] = static_features[i]
            seq2seq_subject_ids[total_seq2seq_rows] = signal_id

            seq2seq_current_treatments[total_seq2seq_rows, prediction_index, :] = seq2seq_current_treatments[total_seq2seq_rows, prediction_index-1, :]
            # TODO previous treatment is nos used yet but we should modify the current index as well
            seq2seq_previous_treatments[total_seq2seq_rows, prediction_index, :] = seq2seq_previous_treatments[total_seq2seq_rows, prediction_index-1, :]


            total_seq2seq_rows += 1

    # Filter everything shorter
    seq2seq_previous_treatments = seq2seq_previous_treatments[:total_seq2seq_rows, :, :]
    seq2seq_current_treatments = seq2seq_current_treatments[:total_seq2seq_rows, :, :]
    seq2seq_static_features = seq2seq_static_features[:total_seq2seq_rows, :]
    seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
    seq2seq_prev_outputs = seq2seq_prev_outputs[:total_seq2seq_rows, :, :]
    seq2seq_vitals = seq2seq_vitals[:total_seq2seq_rows, :, :]
    seq2seq_next_vitals = seq2seq_next_vitals[:total_seq2seq_rows, :, :]
    seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
    seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]

    if 'stabilized_weights' in mimic_dataset.data:
        seq2seq_stabilized_weights = seq2seq_stabilized_weights[:total_seq2seq_rows]

    new_data = {
        'prev_treatments': seq2seq_previous_treatments,
        'current_treatments': seq2seq_current_treatments,
        'static_features': seq2seq_static_features,
        'prev_outputs': seq2seq_prev_outputs,
        'outputs': seq2seq_outputs,
        'vitals': seq2seq_vitals,
        'next_vitals': seq2seq_next_vitals,
        'unscaled_outputs': seq2seq_outputs * mimic_dataset.scaling_params['output_stds'] + mimic_dataset.scaling_params['output_means'],
        'sequence_lengths': seq2seq_sequence_lengths,
        'active_entries': seq2seq_active_entries,
        'subject_ids': seq2seq_subject_ids,
        # 'future_past_split': seq2seq_sequence_lengths - projection_horizon
        'future_past_split': seq2seq_sequence_lengths - time_shift - 1
    }
    if 'stabilized_weights' in mimic_dataset.data:
        new_data['stabilized_weights'] = seq2seq_stabilized_weights

    mimic_dataset.data = new_data

def forecast_ate_values(test_dl: DataLoader, model: torch.nn.Module, s_mean, s_std, time_shift, model_name):
    left_values = []
    right_values = []
    subject_ids = []
    hours_in = []
    left_prices = []
    right_prices = []
    device = model.device
    with torch.no_grad():
        for batch in tqdm(test_dl, desc=f"forecasting model {model_name}"):
            tau_values = batch["future_past_split"].numpy()
            batch_subject_ids = batch["subject_ids"].numpy()
            
            assert batch_subject_ids[0] == batch_subject_ids[1]
            assert tau_values[0] == tau_values[1]
            tau = int(tau_values[0])
            prediction_index = tau+time_shift
            subject_id = batch_subject_ids[0]
            indexes = torch.tensor([tau, tau])

            for key in [
                "vitals", "static_features", "current_treatments", "outputs", "prev_treatments", "prev_outputs", "active_entries"
            ]:
                batch[key] = batch[key].to(device)

            output = model.forecast(batch)
            right_values.append(output[0, prediction_index, 0].item())
            left_values.append(output[1, prediction_index, 0].item())
            subject_ids.append(subject_id)
            hours_in.append(prediction_index)
            right_price, left_price = batch["current_treatments"][0, prediction_index, :].cpu().numpy(), batch["current_treatments"][1, prediction_index, :].cpu().numpy() # vaso / vent | 2vent + vaso
            left_price = 2*left_price[0] + left_price[1]  # 2*vaso + vent
            right_price = 2*right_price[0] + right_price[1]
            left_prices.append(left_price)
            right_prices.append(right_price)   

    left_values = np.array(left_values) * s_std + s_mean
    right_values = np.array(right_values) * s_std + s_mean
    ate_values = right_values - left_values

    return pd.DataFrame({
        "subject_id": subject_ids,
        "hours_in": hours_in,
        "left_predicted_demand": left_values,
        "right_predicted_demand": right_values,
        "ate": ate_values,
        "left_price": left_prices,
        "right_price": right_prices,
    }).astype({"subject_id": int, "hours_in": int})


def compute_rdd_metrics_for_seed(seed, tft_models_dict, device):
    print(f"loading test dataset for seed {seed}")
    # process datasets
    #   process tft datasets
    dataset_collection = MIMIC3RealDatasetCollection(
        "data/processed/all_hourly_data.h5",
        min_seq_length=30,
        max_seq_length=60,
        seed=seed,
        max_number=10000,
        split = {"val":0.15,"test":0.15}, 
        projection_horizon=5,
        autoregressive=True,
        outcome_list = config["outcome_list"],
        vitals = config["vital_list"],
        treatment_list = config["treatment_list"],
        static_list = config["static_list"]
    )
    process_data_rdd(dataset_collection.test_f, time_shift=TIME_SHIFT)

    test_dataset_subject_ids = set(dataset_collection.test_f.data["subject_ids"].tolist())
    seed_rdd_df = rdd_dataset.query("subject_id in @test_dataset_subject_ids")
    print(len(seed_rdd_df))
    s_mean, s_std = dataset_collection.test_f.scaling_params["output_means"], dataset_collection.test_f.scaling_params["output_stds"] 
    
    for model_name, (model_class, model_path) in tqdm(tft_models_dict.items(), desc=f"predictions for seed {seed}"):
        # loading model
        is_ct_model = model_name.startswith("CT")
        if is_ct_model:
            model = model_class(seed=seed, seed_idx=SEEDS.index(seed))
        else:
            model = model_class.load_from_checkpoint(model_path, map_location=device)
            if hasattr(model, "training_m_e"):
                model.training_m_e = False
        model = model.eval()
        model.freeze()
        model = model.to(device)

        # creating dataloader
        test_dl = DataLoader(
            # dataset_collection_ct.test_f if is_ct_model else dataset_collection.test_f, 
            dataset_collection.test_f, 
            batch_size=2,
            shuffle=False
        ) # always use a batchsize of 2 !!!

        # predicting values
        ate_df = forecast_ate_values(
            test_dl=test_dl, 
            model=model, 
            s_mean=s_mean, 
            s_std=s_std, 
            time_shift=TIME_SHIFT,
            model_name=model_name
        )

        seed_rdd_df = seed_rdd_df.merge(
            ate_df[[
                "subject_id", "hours_in", "left_price", "right_price", "left_predicted_demand", "right_predicted_demand"
            ]].rename(columns={
                "left_price": model_name + "_left_price",
                "right_price": model_name + "_right_price",
                "left_predicted_demand": model_name + "_left_predicted_demand",
                "right_predicted_demand": model_name + "_right_predicted_demand",
            }),
            on=["subject_id", "hours_in"],
            how="inner"
        )
        print(len(seed_rdd_df))
        seed_rdd_df[model_name + "_delta_demand"] = seed_rdd_df[model_name + "_right_predicted_demand"] - seed_rdd_df[model_name + "_left_predicted_demand"]

    seed_rdd_df["rdd_delta"] = seed_rdd_df["right_predicted_demand"] - seed_rdd_df["left_predicted_demand"]
    return seed, seed_rdd_df


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    device = torch.device("cuda")
    # device = torch.device("cpu")
    with open('/home/thomas/mimic/physionet.org/files/mimiciii/CausalTransformer/config/dataset/mimic3_real.yaml', 'r') as file:
        config = yaml.safe_load(file)["dataset"]

    
    # fetch checkpoint paths
    tft_models_dict_per_seed = {}
    rdd_df_per_seed = {}
    for model_name_prefix, folder_name_prefix, model_class, seeds in TFT_MODELS:
        for seed, file_index in seeds:
            print("formatting ", f"{folder_name_prefix}_{file_index}")
            print(os.path.join(TFT_MODELS_PREFIX_PATH, f"{folder_name_prefix}_{file_index}", "checkpoints/*.ckpt"))
            print(os.path.isdir(os.path.join(TFT_MODELS_PREFIX_PATH, f"{folder_name_prefix}_{file_index}")))
            tft_models_dict_per_seed.setdefault(seed, {})[f"{model_name_prefix}_{file_index}"] = (
                model_class,
                glob(os.path.join(TFT_MODELS_PREFIX_PATH, f"{folder_name_prefix}_{file_index}", "checkpoints/*.ckpt"))[-1],
            )

    if COMPUTE_CT_VALUES:
        # add CT models
        for seed_index, seed in enumerate(SEEDS):
            tft_models_dict_per_seed.setdefault(seed, {})[f"CT_{seed_index}"] = (
                lambda seed, seed_idx: load_ct_model(seed=seed, seed_idx=seed_idx, device=device),
                None
            )


    print("Reading RDD dataset")
    rdd_dataset = pd.read_parquet(RDD_PATH, columns=[
        "subject_id", "hours_in", "left_days", "right_days", "left_price", "right_price", "left_predicted_demand", "right_predicted_demand", "method"
    ]).query("method in (@RDD_METHODS)").astype({"subject_id": int, "hours_in": int})

    results = joblib.Parallel(n_jobs=N_JOBS, backend='loky')(
        joblib.delayed(compute_rdd_metrics_for_seed)(
            seed=seed,
            tft_models_dict=tft_models_dict,
            device=device
        ) 
        for seed, tft_models_dict in tft_models_dict_per_seed.items()
    )

    for seed, seed_rdd_df in results:
        rdd_df_per_seed[seed] = seed_rdd_df
        seed_rdd_df.to_parquet(os.path.join(TARGET_PATH, f"rdd_metrics_seed_{seed}_time_{TIME_SHIFT}_{METRICS_FILE_SUFFIX}.parquet"))


    # compute metrics per seed
    metrics_dict = {
        "TIME_SHIFT": TIME_SHIFT,
        "RDD_PATH": RDD_PATH,
        "RDD_METHODS": RDD_METHODS,
        "rdd_dataset_len_per_seed": {
            seed: len(rdd_df_per_seed[seed].query(f"method == '{RDD_METHODS[0]}'"))
            for seed in SEEDS
        },
        "rdd_dataset_filtering": {
            "query_str": RDD_QUERY_STR,
            "top_percent_outliers_filtereing": TOP_PERCENT_OUTLIERS_SECLECTION,
            "total_filtering_percent": {
                seed: rdd_df_per_seed[seed].query(f"method == '{RDD_METHODS[0]}'").eval(RDD_QUERY_STR).mean() - (TOP_PERCENT_OUTLIERS_SECLECTION*2)
                for seed in SEEDS
            },
        },
        "models_per_seed": {
            seed:{
                model_name: model_path
                for model_name, (_, model_path)
                in models_per_seed_dict.items() if "CT_" not in model_name
            }
            for seed, models_per_seed_dict in tft_models_dict_per_seed.items()
        },
        "metrics_per_method": {
            method: {
                "raw_metrics":{
                    "metrics_per_seed": {}, 
                    "metrics_per_architecture": {}, 
                },
                "filtered_metrics": {
                    "metrics_per_seed": {},
                    "metrics_per_architecture": {}
                },
            } for method in RDD_METHODS
        }
    }
    for method in RDD_METHODS:
        for seed, tft_models_dict in tft_models_dict_per_seed.items():
            method_df = rdd_df_per_seed[seed].query("method == @method")
            for model_name in tft_models_dict.keys():
                metrics_dict["metrics_per_method"][method]["raw_metrics"]["metrics_per_seed"].setdefault(seed, {}).setdefault("rmse", {})[model_name] = (
                    np.sqrt(mean_squared_error(y_true=method_df["rdd_delta"], y_pred=method_df[model_name + "_delta_demand"]))
                )
                metrics_dict["metrics_per_method"][method]["raw_metrics"]["metrics_per_seed"][seed].setdefault("mae", {})[model_name] = (
                    np.sqrt(mean_absolute_error(y_true=method_df["rdd_delta"], y_pred=method_df[model_name + "_delta_demand"]))
                )
                if RDD_QUERY_STR:
                    filtered_df = method_df.query(RDD_QUERY_STR)
                    if TOP_PERCENT_OUTLIERS_SECLECTION:
                        q1, q2 = filtered_df["rdd_delta"].quantile(q=[TOP_PERCENT_OUTLIERS_SECLECTION, 1-TOP_PERCENT_OUTLIERS_SECLECTION]).values.tolist()
                        filtered_df.query("@q1 <= rdd_delta <= @q2", inplace=True)

                    metrics_dict["metrics_per_method"][method]["filtered_metrics"]["metrics_per_seed"].setdefault(seed, {}).setdefault("rmse", {})[model_name] = (
                        np.sqrt(mean_squared_error(y_true=filtered_df["rdd_delta"], y_pred=filtered_df[model_name + "_delta_demand"]))
                    )
                    metrics_dict["metrics_per_method"][method]["filtered_metrics"]["metrics_per_seed"][seed].setdefault("mae", {})[model_name] = (
                        np.sqrt(mean_absolute_error(y_true=filtered_df["rdd_delta"], y_pred=filtered_df[model_name + "_delta_demand"]))
                    )


        # format metrics per architecture
        metrics_weights = list(metrics_dict["rdd_dataset_len_per_seed"].values())
        metrics_filtered_weights = [
            metrics_dict["rdd_dataset_filtering"]["total_filtering_percent"][SEEDS[idx]] * rdd_dataset_len
            for idx, rdd_dataset_len in enumerate(metrics_weights)
        ]
        for architecture, *_, seeds in TFT_MODELS:
            rmse_metric_values = np.round([
                metrics_dict["metrics_per_method"][method]["raw_metrics"]["metrics_per_seed"][seed]["rmse"][f"{architecture}_{idx}"]
                for seed, idx in seeds
            ], 8).tolist()
            mae_metric_values = np.round([
                metrics_dict["metrics_per_method"][method]["raw_metrics"]["metrics_per_seed"][seed]["mae"][f"{architecture}_{idx}"]
                for seed, idx in seeds
            ], 8).tolist()
            rmse_weighted_average = np.average(rmse_metric_values, weights=metrics_weights)
            mae_weighted_average = np.average(mae_metric_values, weights=metrics_weights)
            metrics_dict["metrics_per_method"][method]["raw_metrics"]["metrics_per_architecture"][architecture] = {
                "rmse":{
                    "values": rmse_metric_values,
                    "mean": np.mean(rmse_metric_values),
                    "std": np.std(rmse_metric_values),
                    "weighted_mean": rmse_weighted_average,
                    "weighted_std": np.round(np.sqrt(np.average((np.array(rmse_metric_values)-rmse_weighted_average)**2, weights=metrics_weights)), 8),
                },
                "mae": {
                   "values": mae_metric_values,
                    "mean": np.mean(mae_metric_values),
                    "std": np.std(mae_metric_values),
                    "weighted_mean": mae_weighted_average,
                    "weighted_std": np.round(np.sqrt(np.average((np.array(mae_metric_values)-mae_weighted_average)**2, weights=metrics_weights)), 8), 
                }
            }
            if RDD_QUERY_STR:
                rmse_metric_values = [
                    metrics_dict["metrics_per_method"][method]["filtered_metrics"]["metrics_per_seed"][seed]["rmse"][f"{architecture}_{idx}"]
                    for seed, idx in seeds
                ]
                mae_metric_values = [
                    metrics_dict["metrics_per_method"][method]["filtered_metrics"]["metrics_per_seed"][seed]["mae"][f"{architecture}_{idx}"]
                    for seed, idx in seeds
                ]
                rmse_weighted_average = np.average(rmse_metric_values, weights=metrics_filtered_weights)
                mae_weighted_average = np.average(mae_metric_values, weights=metrics_filtered_weights)
                metrics_dict["metrics_per_method"][method]["filtered_metrics"]["metrics_per_architecture"][architecture] = {
                    "rmse": {
                        "values": rmse_metric_values,
                        "mean": np.mean(rmse_metric_values),
                        "std": np.std(rmse_metric_values),
                        "weighted_mean": rmse_weighted_average,
                        "weighted_std": np.round(np.sqrt(np.average((np.array(rmse_metric_values)-rmse_weighted_average)**2, weights=metrics_filtered_weights)), 8),
                    },
                    "mae": {
                        "values": mae_metric_values,
                        "mean": np.mean(mae_metric_values),
                        "std": np.std(mae_metric_values),
                        "weighted_mean": mae_weighted_average,
                        "weighted_std": np.round(np.sqrt(np.average((np.array(mae_metric_values)-mae_weighted_average)**2, weights=metrics_filtered_weights)), 8),
                    }
                }
        if COMPUTE_CT_VALUES:
            ct_rmse_metric_values = [
                metrics_dict["metrics_per_method"][method]["raw_metrics"]["metrics_per_seed"][seed]["rmse"][f"CT_{idx}"]
                for idx, seed in enumerate(SEEDS)
            ]
            ct_mae_metric_values = [
                metrics_dict["metrics_per_method"][method]["raw_metrics"]["metrics_per_seed"][seed]["mae"][f"CT_{idx}"]
                for idx, seed in enumerate(SEEDS)
            ]
            ct_rmse_weighted_average = np.average(ct_rmse_metric_values, weights=metrics_weights)
            ct_mae_weighted_average = np.average(ct_mae_metric_values, weights=metrics_weights)
            metrics_dict["metrics_per_method"][method]["raw_metrics"]["metrics_per_architecture"]["CT"] = {
                "rmse": {
                    "values": ct_rmse_metric_values,
                    "mean": np.mean(ct_rmse_metric_values),
                    "std": np.std(ct_rmse_metric_values),
                    "weighted_mean": ct_rmse_weighted_average,
                    "weighted_std": np.round(np.sqrt(np.average((np.array(ct_rmse_metric_values)-ct_rmse_weighted_average)**2, weights=metrics_weights)), 8),
                },
                "mae": {
                    "values": ct_mae_metric_values,
                    "mean": np.mean(ct_mae_metric_values),
                    "std": np.std(ct_mae_metric_values),
                    "weighted_mean": ct_mae_weighted_average,
                    "weighted_std": np.round(np.sqrt(np.average((np.array(ct_mae_metric_values)-ct_mae_weighted_average)**2, weights=metrics_weights)), 8),
                }
            }
            if RDD_QUERY_STR:
                ct_rmse_metric_values = [
                    metrics_dict["metrics_per_method"][method]["filtered_metrics"]["metrics_per_seed"][seed]["rmse"][f"CT_{idx}"]
                    for idx, seed in enumerate(SEEDS)
                ]
                ct_mae_metric_values = [
                    metrics_dict["metrics_per_method"][method]["filtered_metrics"]["metrics_per_seed"][seed]["mae"][f"CT_{idx}"]
                    for idx, seed in enumerate(SEEDS)
                ]
                ct_rmse_weighted_average = np.average(ct_rmse_metric_values, weights=metrics_filtered_weights)
                ct_mae_weighted_average = np.average(ct_mae_metric_values, weights=metrics_filtered_weights)
                metrics_dict["metrics_per_method"][method]["filtered_metrics"]["metrics_per_architecture"]["CT"] = {
                    "rmse": {
                        "values": ct_rmse_metric_values,
                        "mean": np.mean(ct_rmse_metric_values),
                        "std": np.std(ct_rmse_metric_values),
                        "weighted_mean": ct_rmse_weighted_average,
                        "weighted_std": np.round(np.sqrt(np.average((np.array(ct_rmse_metric_values)-ct_rmse_weighted_average)**2, weights=metrics_filtered_weights)), 8),
                    },
                    "mae": {
                        "values": ct_mae_metric_values,
                        "mean": np.mean(ct_mae_metric_values),
                        "std": np.std(ct_mae_metric_values),
                        "weighted_mean": ct_mae_weighted_average,
                        "weighted_std": np.round(np.sqrt(np.average((np.array(ct_mae_metric_values)-ct_mae_weighted_average)**2, weights=metrics_filtered_weights)), 8),
                    }
                    
                }


    # compute metrics per with filtered
    with open(os.path.join(TARGET_PATH, f"metrics_{TIME_SHIFT}_{METRICS_FILE_SUFFIX}.json"), "w") as f:
        f.truncate(0)
        f.seek(0)
        json.dump(metrics_dict, f, indent=4)

    print(f"METRICS FOR TIME SHIFT {TIME_SHIFT}")
    print(json.dumps(metrics_dict, indent=4))

        