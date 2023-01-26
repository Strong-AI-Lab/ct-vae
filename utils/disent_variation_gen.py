
import torch
from torch.utils.data import random_split

from disent.dataset.data import Cars3dData, DSpritesData, DSpritesImagenetData, Mpi3dData, SmallNorbData, Shapes3dData, SpritesData, XYSquaresData, XYObjectData, XYObjectShadedData
# from disent.dataset import DisentDataset

import pandas as pd
from collections import Counter
import random
from tqdm import tqdm


# Build dataset
DATASETS = {
        "cars3d": Cars3dData,
        "dsprites": DSpritesData,
        # "dSpritesImageNet": DSpritesImagenetData,
        # "mpi3d": Mpi3dData,
        "smallnorb": SmallNorbData,
        "3dshapes": Shapes3dData,
        "sprites": SpritesData,
        # "XYSquares": XYSquaresData,
        # "XYObject": XYObjectData,
        # "XYObjectShaded": XYObjectShadedData
    }
dataset_name = "3dshapes"
data = DATASETS[dataset_name](data_root="Data/",prepare=True)
# dataset = DisentDataset(data, return_factors=True)


# Save splits
data_len = len(data)
split_coeffs = [int(round(l*c)) for l, c in zip([data_len]*3,[0.7,0.15,0.15])]
split_coeffs[0] = data_len - sum(split_coeffs[1:])
splits = random_split(range(data_len), split_coeffs, generator=torch.Generator().manual_seed(42))

split_list = [0] * data_len
for i,s in enumerate(tqdm(splits)):
    for id in s:
        split_list[id] = i

s_df = pd.DataFrame({
    'Id': range(data_len),
    'Split': split_list,
})
print("Split dataframe generated.", Counter(split_list))
s_df.to_csv(f'Data/{dataset_name}/list_eval_partition.txt')


# Save variations
max_variations = len(data.factor_names)
max_instances_per_variation = 1000
print("Factors : ", *zip(data.factor_names, data.factor_sizes))
inputs = []
outputs = []
variations = []
sources = []
targets = []
vsplits = []

uniques = set()
for f, _ in enumerate(tqdm(data.factor_names)):
    for v in tqdm(range(data.factor_sizes[f] - 1)):
        samples = [[random.sample(range(i),k=1)[0] for i in data.factor_sizes] for _ in range(max_instances_per_variation)]
        for sample in tqdm(samples):
            s = v
            t = v+1
            pos_s = sample.copy()
            pos_t = sample.copy()
            pos_s[f] = s
            pos_t[f] = t
            
            inp = data.pos_to_idx(pos_s)
            out = data.pos_to_idx(pos_t)
            if split_list[inp] == split_list[out] and (inp, out) not in uniques and (out, inp) not in uniques:
                inputs += [inp, out]
                outputs += [out, inp]
                variations += [f, f]
                sources += [s, t]
                targets += [t, s]
                vsplits += [split_list[inp],split_list[inp]] 
                uniques.add((inp, out))
                uniques.add((out, inp))

df = pd.DataFrame({
    'Inputs': inputs,
    'Outputs': outputs,
    'Variations': variations,
    'Sources': sources,
    'Targets': targets,
    'Splits': vsplits,
})
print("Variations: ", Counter(variations), "Splits: ", Counter(vsplits))
print("Number of duplicates: ", len(inputs) - len(uniques))
print("Variation dataframe generated.")
print(df)

df.to_csv(f'Data/{dataset_name}/variation_attrs_{max_variations}.txt')