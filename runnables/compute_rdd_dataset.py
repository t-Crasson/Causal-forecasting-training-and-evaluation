import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from src.rdd.rdd import compute_rdd_values_n_jobs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_name="config.yaml", config_path="../config/")
def main(args: DictConfig):
    """
    Computing the CATE dataset using RDD method
    :param args: arguments of run as DictConfig

    Returns: path of the saved dataframe
    """

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    # Format a dataframe from the dataset
    logger.info("Reading and formatting dataframe")
    df = pd.read_parquet("/home/thomas/mimic/physionet.org/files/mimiciii/causal-forecasting-bis/rdd/data/mimic_raw_treatment_dataset_all_seeds.parquet")
    df["treatment"] = 2*df["vaso"] + df["vent"]

    # Compute the RDD dataset
    logger.info("Compute RDD dataset")
    rdd_df = compute_rdd_values_n_jobs(
        df=df,
        treatment_column=args.rdd.treatment_column,
        outcome_column=args.rdd.outcome_column,
        rdd_model_class_path=args.rdd.rdd_model_class_path,
        time_step_column=args.rdd.time_step_column,
        time_series_unique_id_columns=args.rdd.time_series_unique_id_columns,
        rdd_model_kwargs=args.rdd.rdd_model_kwargs,
        static_columns_to_add=args.rdd.static_columns_to_add,
        n_jobs=args.rdd.n_jobs,
    )

    rdd_df.to_parquet(args.destination_file_path)
    logger.info(f"Saved dataframe at {args.destination_file_path}")

    return args.destination_file_path
