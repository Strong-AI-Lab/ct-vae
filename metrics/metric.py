
from disent.metrics import metric_dci, metric_mig, metric_sap, metric_factor_vae
from disent.dataset import DisentDataset

from typing import Callable, List


METRICS = {
    "DCI" : metric_dci,
    "MIG" : metric_mig,
    "SAP" : metric_sap,
    "FactorVaeScore" : metric_factor_vae,
    "": None
}


class Metric():
    
    def __init__(self, 
                    metric_name: str, 
                    dataset : DisentDataset, 
                    batch_size: int = 64,
                    num_train: int = 1000,
                    num_test: int = 500,
                    **kwargs):
        self.metric = METRICS[metric_name]
        self.name = metric_name
        self.dataset = dataset
        self.args = {
            "batch_size": batch_size,
            "num_train": num_train,
            "num_test": num_test,
        }

        if metric_name == "MIG":
            del self.args["num_test"]

        if metric_name == "FactorVaeScore":
            del self.args["num_test"]
            self.args["num_eval"] = num_test
            self.args["num_variance_estimate"] = 64*2**3

    def compute(self, repr_func: Callable):
        return self.metric(self.dataset, repr_func, **self.args)


class MetricSet(Metric):

    def __init__(self, 
                    metric_names: List[str], 
                    dataset : DisentDataset, 
                    batch_size: int = 64,
                    num_train: int = 1000,
                    num_test: int = 500,
                    **kwargs):
        self.metrics = [Metric(name, dataset, batch_size, num_train, num_test) for name in metric_names]

    def compute(self, repr_func: Callable):
        res = {}
        for metric in self.metrics:
            val = metric.compute(repr_func)
            # res[metric.name] = val
            res = {**res, **val}
        
        return res