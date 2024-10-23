import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import os
import gc
import torch

from src.data.mimic_iii.real_dataset import MIMIC3RealDatasetCollectionCausal
from src.models.utils import set_seed
from src.rdd.utils import from_fully_qualified_import


import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
import logging
from hydra.core.hydra_config import HydraConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_m0(
    args: DictConfig, 
    dataset_collection: MIMIC3RealDatasetCollectionCausal,
    splitted_directory: list[str],
    seed_idx: int
):
    set_seed(args.exp.seed)

    model_kwargs = dict(args.model.params)
    model_kwargs["treatment_module_class"] = from_fully_qualified_import(args.model.params.treatment_module_class)

    model_class = from_fully_qualified_import(args.model._target_)
    model = model_class(**model_kwargs)

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=args.exp.max_epochs,        
        devices=args.exp.gpus,
        callbacks=ModelCheckpoint(
            filename='{epoch}-{val_loss:.2f}',
            monitor = "val_loss",
            mode="min"
        ),
        logger=TensorBoardLogger(
            save_dir=os.path.sep.join(splitted_directory[:-1]), 
            name=splitted_directory[-1],
            version=f"m_e_{args.model.name}_{seed_idx}"
        ),
        deterministic=args.exp.deterministic,
    )

    train_loader_s1 = DataLoader(dataset_collection.train_f_multi_s1,shuffle=True,batch_size=args.dataset.batch_size)
    val_loader_s1 = DataLoader(dataset_collection.val_f_multi_s1, shuffle=False,batch_size=512)
    model.training_theta = False
    model.train()
    trainer.fit(model, train_loader_s1, val_loader_s1)

    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()


def train_theta(
    args: DictConfig, 
    dataset_collection: MIMIC3RealDatasetCollectionCausal,
    splitted_directory: list[str],
    seed_idx: int
):
    set_seed(args.exp.seed)

    model_class = from_fully_qualified_import(args.model._target_)
    m_e_model_path = os.path.join(args.model.destination_directory, f"m_e_{args.model.name}_{seed_idx}", "checkpoints")
    model = model_class.load_from_checkpoint(os.path.join(m_e_model_path, os.listdir(m_e_model_path)[0])).to("cuda")

    for theta_param, theta_param_value in dict(args.model.theta_params).items():
        setattr(model, theta_param, theta_param_value)
        setattr(model.hparams, theta_param, theta_param_value)
    model.training_theta = True
    model.save_hyperparameters()

    model.train()
    model.configure_optimizers()

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=args.exp.theta_max_epochs,
        devices=args.exp.gpus,
        callbacks=ModelCheckpoint(
            filename='{epoch}-{val_loss:.2f}',
            monitor = "val_loss",
            mode="min"
        ),
        logger=TensorBoardLogger(
            save_dir=os.path.sep.join(splitted_directory[:-1]), 
            name=splitted_directory[-1],
            version=f"theta_{args.model.name}_{seed_idx}"
        ),
        deterministic=args.exp.deterministic,
    )

    train_loader_s2 =  DataLoader(dataset_collection.train_f_multi_s2, shuffle=True, batch_size=args.dataset.batch_size)
    val_loader_s2 = DataLoader(dataset_collection.val_f_multi_s2, shuffle=False,batch_size=512)

    trainer.fit(model, train_loader_s2, val_loader_s2)


@hydra.main(config_name="config.yaml", config_path="./config/", version_base="1.3.2")
def main(args: DictConfig):

    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda *args: sum(list(args)), replace=True)
    OmegaConf.register_new_resolver("len", len, replace=True)
    logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))

    dataset_collection = MIMIC3RealDatasetCollectionCausal(
        args.dataset.path,
        min_seq_length=args.dataset.min_seq_length,
        max_seq_length=args.dataset.max_seq_length,
        seed=args.exp.seed,
        max_number=args.dataset.max_number,
        split=args.dataset.split,
        projection_horizon=args.dataset.projection_horizon,
        autoregressive=args.dataset.autoregressive,
        outcome_list=args.dataset.outcome_list,
        vitals=args.dataset.vital_list,
        treatment_list=args.dataset.treatment_list,
        static_list=args.dataset.static_list,
        split_causal={"S1": 0.5}
    )

    dataset_collection.process_data_multi_val()
    dataset_collection.process_data_multi_train()

    splitted_directory = args.model.destination_directory.split(os.path.sep)
    try:
        seed_idx = HydraConfig.get().job.num
    except MissingMandatoryValue:
        seed_idx = 0
    
    logger.info("Training m0/e0")
    train_m0(args, dataset_collection, splitted_directory, seed_idx)
    logger.info("Training theta")
    train_theta(args, dataset_collection, splitted_directory, seed_idx)

    

if __name__=="__main__":
    main()
        
