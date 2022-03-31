from tops.config import LazyCall as L
# The line belows inherits the configuration set for the tdt4265 dataset
from .tdt4265 import (
    train,
    optimizer,
    anchors,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map
)

train_cpu_transform