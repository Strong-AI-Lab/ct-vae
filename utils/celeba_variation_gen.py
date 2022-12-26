
import re
import random
import pandas as pd
from collections import Counter

# Open file
file = open("Data/celeba/list_attr_celeba.txt", "r")
f_split = open("Data/celeba/list_eval_partition.txt", "r")
nb_images, labels, *lines = file.readlines()


# Generate variation hashes
max_variations = 40
max_instances_per_variation = 2
labels = re.split("\s+", labels)[:-1][:max_variations]
hashes = {}

for l in lines:
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
for l in f_split.readlines():
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
for k, h in hashes.items():
    v = h["variation"]
    h_neg = random.choices(h["neg"], k=min(max_instances_per_variation, len(h["neg"])))
    h_pos = random.choices(h["pos"], k=min(max_instances_per_variation, len(h["pos"])))
    for img_neg in h_neg:
        for img_pos in h_pos:
            if img_splits[img_pos] == img_splits[img_neg]:
                inputs += [img_neg, img_pos]
                outputs += [img_pos, img_neg]
                variations += [v, v]
                sources += [-1, 1]
                targets += [1, -1]
                splits += [img_splits[img_pos], img_splits[img_pos]]
print("Rows generated.", Counter(splits))


df = pd.DataFrame({
    'Inputs': inputs,
    'Outputs': outputs,
    'Variations': variations,
    'Sources': sources,
    'Targets': targets,
    'Splits': splits,
})
print("Dataframe generated.")


# Save file
df.to_csv(f'Data/celeba/variation_attrs_{max_variations}.txt')



