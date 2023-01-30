
# Causal Transition VAE


Repository for the CT-VAE model. This repository is based on a fork of the [pytorch-VAE](https://github.com/AntixK/PyTorch-VAE) library with two new models: the MCQ-VAE and the CT-VAE, and additional controls over the datasets and experiments.


## Datasets

The repository contains several datasets for reconstruction tasks:
  <!-- - [AFHQ](https://openaccess.thecvf.com/content_CVPR_2020/html/Choi_StarGAN_v2_Diverse_Image_Synthesis_for_Multiple_Domains_CVPR_2020_paper.html) -->
  - [CelebA](https://openaccess.thecvf.com/content_iccv_2015/html/Liu_Deep_Learning_Face_ICCV_2015_paper.html)
  - [Cars3D](https://papers.nips.cc/paper/2015/hash/e07413354875be01a996dc560274708e-Abstract.html)
  - [DSprites](https://openreview.net/forum?id=Sy2fzU9gl) 
  - [SmallNORB](https://www.computer.org/csdl/proceedings-article/cvpr/2004/215820097/12OmNwOnn1p)
  - [Shapes3D](http://proceedings.mlr.press/v80/kim18b.html)
  - [Sprites](https://proceedings.neurips.cc/paper/2015/hash/e07413354875be01a996dc560274708e-Abstract.html)

The CelebA dataset can be downloaded [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) while the other disentanglement datasets are obtained using the [disent](https://github.com/nmichlo/disent) library. For installing CelebA, please follow the recommendations from the original Pytorch-VAE repository.


Variants of these datasets for Causal Transition tasks are available: 
    
  - TCeleba
  <!-- - TAFHQ -->
  - TSprites
  - TShapes3D
  - TSmallNORB
  - TDSprites
  - TCars3D

The transition datasets contain pairs of images such that the value of only one label changes between the two images. If the label is categorical, then a transition can happen only between two adjacent values.

To create the transition datasets, run the following scripts:
```
$ python utils/celeba_variation_gen.py
$ python utils/disent_variation_gen.py <dataset_name>
```


## Installation
```
$ git clone https://github.com/Strong-AI-Lab/ct-vae
$ cd PyTorch-VAE
$ pip install -r requirements.txt
```

## Usage

### Run experiments

To run experiments, use the following lines of code:
```
$ cd PyTorch-VAE
$ python run.py -c configs/<config-file-name.yaml>
```

### Config file

```yaml
model_params:
  name: "<name of VAE model>"
  in_channels: <number of channels in the image, e.g. 3 for colour images and 1 for B&W>
    .         # Other parameters required by the model
    .
    .

data_params:
  data_path: "<path to the dataset storage, 'Data/' by default>"
  dataset_name: "<name of the dataset>"
  train_batch_size: 64
  val_batch_size:  64
  patch_size: 64
  num_workers: 4
  limit: <restriction to the size of the dataset, avalaible for TDatasets only>
  distributed: <True if using several GPUs and False otherwise, this parameter is needed for TDatasets only>
  
exp_params:
  manual_seed: 1265
  LR: 0.005
  find_unused_parameters: <True if model does not train all its parameters during a forward pass, False otherwise>
  update_parameters: <subset of parameters to train, if specified, freezes the training of all other parameters of the model>
    .         # Other arguments required for training, like scheduler etc.
    .
    .

trainer_params:
  gpus: 1         
  max_epochs: 100
  gradient_clip_val: 1.5
  resume_from_checkpoint: "<optional, path to the model checkpoint to to load the model from>"
  load_weights_only: <use only if 'resume_from_checkpoint' is specified, if True, will not load the state of the optimizers>
    .
    .
    .

logging_params:
  save_dir: "logs/"
  name: "<experiment name>"
```

### Logs

Tensorboard logs can be accessed here:
```
$ cd logs/<experiment name>/version_<the version you want>
```
The experiments also store logs with [wandb](https://wandb.ai/site).

### Hyperparameter search

This repository allows hyperparameter search using [ray tune](https://www.ray.io/ray-tune):

```
$ cd PyTorch-VAE
$ python hyperparameter_search.py -c configs_hyp/<config-file-name.yaml>
```

## License

TODO


## Citations

If you use this repository, please cite:
TODO