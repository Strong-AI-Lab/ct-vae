
import re
import random
import pandas as pd
from collections import Counter
from tqdm import tqdm

# Open file
file = open("Data/celeba/list_attr_celeba.txt", "r")
f_split = open("Data/celeba/list_eval_partition.txt", "r")
nb_images, labels, *lines = file.readlines()


# Generate variation hashes
max_variations = 10
max_instances_per_variation = 8
labels = re.split("\s+", labels)[:-1][:max_variations]
hashes = {}

for l in tqdm(lines):
    img_name, *attrs = re.split("\s+", l)
    attrs = [int(attr) for attr in attrs if re.match(".?\d+", attr)][:max_variations]
    for variation in range(len(attrs)):
        attrs_v = attrs.copy()
        attrs_v[variation] = "?"
        attrs_h = str(attrs_v)
        if attrs_h not in hashes:
            hashes[attrs_h] = {
                            # "variation": labels[variation],
                            "variation": variation,
                            "neg": [],
                            "pos": []
                        }
        mode = "pos" if attrs[variation] == 1 else "neg"
        hashes[attrs_h][mode] += [img_name]
print("Variables generated.")


# Build train/eval/test split
img_splits = {}
for l in tqdm(f_split.readlines()):
    img_name, s, *_ = re.split("\s+", l)
    img_splits[img_name] = int(s)
print("Split generated.")


# Transform to dataframe
inputs = []
outputs = []
variations = []
sources = []
targets = []
splits = []
uniques = set()
for k, h in tqdm(hashes.items()):
    v = h["variation"]
    h_neg = random.choices(h["neg"], k=min(max_instances_per_variation, len(h["neg"])))
    h_pos = random.choices(h["pos"], k=min(max_instances_per_variation, len(h["pos"])))
    for img_neg in h_neg:
        for img_pos in h_pos:
            if img_splits[img_pos] == img_splits[img_neg] and (img_neg, img_pos) not in uniques and (img_pos, img_neg) not in uniques:
                inputs += [img_neg, img_pos]
                outputs += [img_pos, img_neg]
                variations += [v, v]
                sources += [0, 1]
                targets += [1, 0]
                splits += [img_splits[img_pos], img_splits[img_pos]]
                uniques.add((img_neg, img_pos))
                uniques.add((img_pos, img_neg))
print("Rows generated.", Counter(splits))


df = pd.DataFrame({
    'Inputs': inputs,
    'Outputs': outputs,
    'Variations': variations,
    'Sources': sources,
    'Targets': targets,
    'Splits': splits,
})
print("Variations: ", Counter(variations), "Splits: ", Counter(splits))
print("Number of duplicates: ", len(inputs) - len(uniques))
print("Dataframe generated.")
print(df)


# Save file
df.to_csv(f'Data/celeba/variation_attrs_{max_variations}.txt')



