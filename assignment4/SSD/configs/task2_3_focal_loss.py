
from tops.config import LazyCall as L
from ssd.modeling import FocalLoss
from .task2_3_fpn import (
    train,
    optimizer,
    anchors,
    schedulers,
    #loss_objective,
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

loss_objective = L(FocalLoss)(anchors="${anchors}", num_classes=model.num_classes, gamma = 2, alpha=[0.01,*[1 for i in range(model.num_classes-1)]])