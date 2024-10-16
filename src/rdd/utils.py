from typing import Any

import pandas as pd

from src import ROOT_PATH
from src.data.mimic_iii.load_data import load_mimic3_data_processed
from tqdm.auto import tqdm


def load_mimic_raw_data(
    seeds: list[int],
    dataset_config: dict[str, Any]
) -> pd.DataFrame:
    total_df = []
    print(dataset_config)
    for seed in tqdm(seeds, desc="Load raw data from seeds"):
        treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params = (
            load_mimic3_data_processed(
                ROOT_PATH + '/' + dataset_config["path"],
                min_seq_length=0,
                max_seq_length=None,
                max_number=dataset_config["max_number"],
                data_seed=seed,
                outcome_list=dataset_config["outcome_list"],
                vitals=dataset_config["vital_list"],
                treatment_list=dataset_config["treatment_list"],
                static_list=dataset_config["static_list"]
            )
        )
        merged_df = treatments.join(outcomes_unscaled).reset_index()
        total_df.append(merged_df)

    return pd.concat(total_df, ignore_index=True).drop_duplicates(["subject_id", "hours_in"])


def encode_treatments(df: pd.DataFrame, treatment_column: str) -> pd.DataFrame:
    df[treatment_column] = 2 * df["vaso"] + df["vent"]
    return df
