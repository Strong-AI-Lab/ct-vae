
import torch
from models import VQVAE, CTMCQVAE, CausalTransition
from torchviz import make_dot
import json


def build_comp_graph(v,max_rec=10):
    if max_rec==0 or v is None:
        return {}
    children = v.next_functions
    res = {}
    for child in children:
        res[str(child)] = build_comp_graph(child[0],max_rec=max_rec-1)
    return res


x=torch.rand((2,3,256,256))
model=CTMCQVAE(3,64,20,16,[64,128,256],img_size=256)
y = model(x)
l=model.loss_function(*y)


dot=make_dot(l["loss"],params=dict(model.named_parameters()),show_attrs=True,show_saved=True)
dot.render(f"images/ct_computation_graph", format="png")