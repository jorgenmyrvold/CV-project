from tops.config import LazyCall as L
import torch
from .utils import get_dataset_dir

from .task2_3_focal_loss import (
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

data_train.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
data_train.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/train_annotations.json")
data_val.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
data_val.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/val_annotations.json")

optimizer = L(torch.optim.AdamW)(
    lr=5e-4, 
    weight_decay=0.0001,
    amsgrad=True
)
