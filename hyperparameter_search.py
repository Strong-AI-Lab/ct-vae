
import yaml
import argparse
from pathlib import Path

from models import *
from metrics import MetricSet
from experiment import VAEXperiment
from dataset import VAEDataset

import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.logger import DEFAULT_LOGGERS
from ray.air.integrations.wandb import WandbLoggerCallback


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.full_load(file)
    except yaml.YAMLError as exc:
        print(exc)


# Create search space
def to_tune(config_id, config):
    if type(config_id) == dict:
        for current_id, nested_id in config_id.items():
            config[current_id] = to_tune(nested_id, config[current_id])
    elif type(config_id) == list:
        for nested_id in config_id:
            config[nested_id] = to_tune(None, config[nested_id])
    else:
        if type(config) == list:
            config = tune.choice(config)
        elif type(config) == tuple:
            config = tune.uniform(config[0], config[1])
    return config

config = to_tune(config["hyperparameter_search"]["params"], config)


# Loggers
wb_logger = WandbLogger(project="CT-VAE",
                        name=config['logging_params']['name'])
                    
# Save hyperparameters
wb_logger.log_hyperparams(config)


# Def training fcuntion 
def hyp_search(config):
    # Build model
    model = vae_models[config['model_params']['name']](**config['model_params'])

    # Data
    data = VAEDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
    data.setup()
    
    # Experiment
    experiment = VAEXperiment(model,
                            None,
                            None,
                            config['exp_params'],
                            val_sampling=False)

    runner = Trainer(logger=[wb_logger],
                    callbacks=[
                        TuneReportCallback({"loss": "val_loss"}, on="validation_end"),
                    ],
                    strategy=DDPPlugin(find_unused_parameters=config['exp_params']['find_unused_parameters']),
                    **config['trainer_params'])

    runner.fit(experiment, datamodule=data)


# Hyperparameter search
analysis = tune.run(
    tune.with_resources(
        hyp_search,
        config["hyperparameter_search"]["resources_per_trial"],
    ),
    config=config,
    num_samples=config["hyperparameter_search"]["num_samples"],
    callbacks=[WandbLoggerCallback(
                project="CT-VAE",
                log_config=True)]
    )

print(analysis)

