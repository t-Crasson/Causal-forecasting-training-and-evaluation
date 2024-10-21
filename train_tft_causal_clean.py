import torch
import lightning.pytorch as lp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import json
import torch.nn as nn
import warnings
import yaml
import numpy as np
import os
import random
import gc

warnings.filterwarnings("ignore")
from pytorch_lightning import seed_everything
from src.models.causal_tft.tft_causal import CausalTFT
from src.models.causal_tft.treatment_encoder import OneHotTreatmentModule, CumulativeTreatmentModule
from src.data.mimic_iii.real_dataset import MIMIC3RealDatasetCollectionCausal, MIMIC3RealDatasetCollection

MODEL_PREFIX_PATH = "/home/thomas/fork_causal_transformer/Causal-forecasting-training-and-evaluation"
MODEL_PREFIX_FOLDER = "TFT_repro_clean"
IS_CDF = False
MODEL_PREFIX_NAME = f"{'cdf' if IS_CDF else 'density'}_clean_ref"
MODEL_PREFIX_NAME = f"{'cdf' if IS_CDF else 'density'}_clean_ref"

if __name__ == "__main__":
    with open('/home/thomas/mimic/physionet.org/files/mimiciii/CausalTransformer/config/dataset/mimic3_real.yaml',
              'r') as file:
        config = yaml.safe_load(file)["dataset"]

    seeds = [10, 101, 1001, 10010, 10110]
    for i in range(len(seeds)):
        torch.manual_seed(seeds[i])
        torch.cuda.manual_seed(seeds[i])
        np.random.seed(seeds[i])
        random.seed(seeds[i])
        torch.backends.cudnn.benchmark = True
        seed_everything(seeds[i], workers=True)
        torch.multiprocessing.set_start_method("spawn", force=True)
        horizon = 60
        tau = 5
        hidden_size = 128
        batch_size = 64
        embedding_size_stat = []
        embedding_size_future = []

        checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.2f}',
                                              monitor="val_loss",
                                              mode="min")
        model = CausalTFT(
            treatment_module_class=CumulativeTreatmentModule if IS_CDF else OneHotTreatmentModule,
            treatment_max_value=2,
            horizon=horizon,
            projection_length=5,
            target_size=1,
            static_features_size=44,
            hidden_size=hidden_size,
            n_heads=8,
            learning_rate=1e-4,
            attn_dropout=0.1 if IS_CDF else .15,
            dropout=0.1 if IS_CDF else .15,
            temporal_features_size=28,
            last_nn=[hidden_size],
            static_embedding_sizes=embedding_size_stat,
            temporal_embedding_sizes=embedding_size_future,
            trend_size=1,
            n_att_layers=1,
            n_static_layers=1,
            conv_blocks=2,
            conv_padding_size=64,
            kernel_size=5,
            weight_decay=1e-2,
        )
        dataset_collection = MIMIC3RealDatasetCollectionCausal(
            "data/processed/all_hourly_data.h5",
            min_seq_length=30,
            max_seq_length=60,
            seed=seeds[i],
            max_number=10000,
            split={"val": 0.15, "test": 0.15},
            projection_horizon=5,
            autoregressive=True,
            outcome_list=config["outcome_list"],
            vitals=config["vital_list"],
            treatment_list=config["treatment_list"],
            static_list=config["static_list"],
            split_causal={"S1": 0.5}
        )
        dataset_collection.process_data_multi_val()
        dataset_collection.process_data_multi_train()

        # logger = TensorBoardLogger(save_dir=MODEL_PREFIX_PATH, name=MODEL_PREFIX_FOLDER,version = f"m_e_{MODEL_PREFIX_NAME}_{i}")
        # trainer = pl.Trainer(accelerator ="gpu",
        #                     #strategy='ddp_find_unused_parameters_true',
        #                     max_epochs = 15,
        #                     devices = 1,
        #                     callbacks = checkpoint_callback,
        #                     logger = logger,
        #                     deterministic=not IS_CDF
        #                     )

        # train_loader_s1 = DataLoader(dataset_collection.train_f_multi_s1,shuffle=True,batch_size=batch_size)
        # val_loader_s1 = DataLoader(dataset_collection.val_f_multi_s1, shuffle=False,batch_size=512)
        # model.training_theta = False
        # model.train()
        # trainer.fit(model,train_loader_s1,val_loader_s1)

        train_loader_s2 = DataLoader(dataset_collection.train_f_multi_s2, shuffle=True, batch_size=batch_size)
        val_loader_s2 = DataLoader(dataset_collection.val_f_multi_s2, batch_size=512)
        # del model

        path = f"{MODEL_PREFIX_PATH}/{MODEL_PREFIX_FOLDER}/m_e_{MODEL_PREFIX_NAME}_{i}/checkpoints"
        file = os.listdir(path)[0]
        path = os.path.join(path, file)
        model = CausalTFT.load_from_checkpoint(path).to("cuda")

        model.weight_decay = 1e-2
        model.learning_rate = 1e-4
        # model.learning_rate = 5e-5

        model.hparams.weight_decay = model.weight_decay
        model.hparams.learning_rate = model.learning_rate
        model.save_hyperparameters()

        model.training_theta = True
        model.train()
        model.configure_optimizers()

        del trainer
        del logger

        torch.cuda.empty_cache()
        gc.collect()

        checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.2f}',
                                              monitor="val_loss",
                                              mode="min")
        logger = TensorBoardLogger(save_dir=MODEL_PREFIX_PATH, name=MODEL_PREFIX_FOLDER,
                                   version=f"theta_{MODEL_PREFIX_NAME}_{i}")
        trainer = pl.Trainer(accelerator="gpu",
                             # strategy='ddp_find_unused_parameters_true',
                             max_epochs=10,
                             devices=1,
                             callbacks=checkpoint_callback,
                             logger=logger,
                             deterministic=not IS_CDF
                             )
        trainer.fit(model, train_loader_s2, val_loader_s2)
        # break

