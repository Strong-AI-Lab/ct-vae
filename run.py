
import os
import yaml
import argparse
import numpy as np
from pathlib import Path

from models import *
from metrics import MetricSet
from experiment import VAEXperiment

import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.strategies import DDPStrategy


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


# Loggers
tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['logging_params']['name'],)

wb_logger = WandbLogger(project="CT-VAE",
                        name=config['logging_params']['name'])
                    
# Save hyperparameters
tb_logger.log_hyperparams(config)
wb_logger.log_hyperparams(config)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)


# Build model
model = vae_models[config['model_params']['name']](**config['model_params'])

# Gradient tracking
wb_logger.watch(model, log_freq=500)  


# Data
data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
data.setup()


# Experiment
train_metric = None
val_metric = None
if "metrics" in config["exp_params"]:
    # train_metric = MetricSet(config["exp_params"]["metrics"],
    #                         data.train_dataset.dataset._full_data,
    #                         batch_size = config["data_params"]["train_batch_size"],
    #                         num_train = config["data_params"]["train_batch_size"] * 20,
    #                         num_test = config["data_params"]["train_batch_size"] * 10)
    val_metric = MetricSet(config["exp_params"]["metrics"],
                            data.val_dataset.dataset._full_data,
                            batch_size = config["data_params"]["val_batch_size"],
                            num_train = config["data_params"]["train_batch_size"] * 20,
                            num_test = config["data_params"]["train_batch_size"] * 10)

experiment = VAEXperiment(model,
                          train_metric,
                          val_metric,
                          config['exp_params'],
                          val_sampling=True,
                          wandb_logger=True)

trainer_config = config["trainer_params"].copy()
if "resume_from_checkpoint" in trainer_config and "load_weights_only" in trainer_config and trainer_config["load_weights_only"]:
    model.load_state_dict({k[6:] : v for k, v in torch.load(trainer_config["resume_from_checkpoint"])["state_dict"].items()}, strict=False) # need to select only model state_dict and remove 'model.' string in keys
    del trainer_config["resume_from_checkpoint"]
    del trainer_config["load_weights_only"]

runner = Trainer(logger=[tb_logger, wb_logger],
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath= os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_Reconstruction_Loss",
                                     save_last= True),
                 ],
                 strategy=DDPStrategy(find_unused_parameters=config['exp_params']['find_unused_parameters']),
                 replace_sampler_ddp = False,
                 **trainer_config)


Path(f"{tb_logger.log_dir}/Inputs").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)