
from tops.config import LazyCall as L
import torch


from .task2_3_w_init import (
    train,
    # optimizer,
    anchors,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    anchors,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map
)


optimizer = L(torch.optim.AdamW)(
    lr=5e-4, 
    weight_decay=0.0001,
    amsgrad=True
)
