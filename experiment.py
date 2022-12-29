import os
import math
import torch
from torch import optim
from models import BaseVAE
from metrics import Metric
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 train_metric: Metric,
                 val_metric: Metric,
                 params: dict,
                 val_sampling: bool = True) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.val_sampling = val_sampling
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels, *args = batch
        self.curr_device = real_img.device

        kwargs = {} if len(args) < 1 or type(args[0]) != dict else args[0]

        results = self.forward(real_img, labels = labels, **kwargs)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)
        train_metric = {} if self.train_metric is None else {key: torch.tensor(val, dtype=torch.float32) for key, val in self.train_metric.compute(self.metric_func).items()}
        
        self.log_dict({**{key: val.item() for key, val in train_loss.items()}, **train_metric}, sync_dist=True, batch_size=real_img.size(0))

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels, *args = batch
        self.curr_device = real_img.device
        
        kwargs = {} if len(args) < 1 or type(args[0]) != dict else args[0]

        results = self.forward(real_img, labels = labels, **kwargs)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
        test_metric = {} if self.val_metric is None else {key: torch.tensor(val, dtype=torch.float32) for key, val in self.val_metric.compute(self.metric_func).items()}
        
        self.log_dict({**{f"val_{key}": val.item() for key, val in val_loss.items()}, **test_metric}, sync_dist=True, batch_size=real_img.size(0))

        
    def on_validation_end(self) -> None:
        if self.val_sampling:
            self.sample_images()
    
    def metric_func(self, x: Tensor) -> Tensor:
        x = x.to(next(self.model.parameters()).device)
        x = self.model.encode(x)[0]
        x = x.view(x.size(0),-1)
        return x
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label, *args = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        kwargs = {} if len(args) < 1 or type(args[0]) != dict else args[0]

        vutils.save_image( test_input.data,
                          os.path.join(self.logger.log_dir , 
                                       "Inputs", 
                                       f"inputs_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label, **kwargs)
        vutils.save_image( recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(32,#144,
                                        self.curr_device,
                                        labels = test_label,
                                        **kwargs)
            vutils.save_image( samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"sample_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
