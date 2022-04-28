
from tops.config import LazyCall as L
from ssd.modeling.retina_net_w_init import RetNetWInit
from ssd.modeling import AnchorBoxes
import torchvision
import torch

from .task2_3_dpr_reg_hds import (
    train,
    optimizer,
    anchors,
    schedulers,
    loss_objective,
    #model,
    backbone,
    data_train,
    data_val,
    anchors,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map
)


model = L(RetNetWInit)(
    feature_extractor = "${backbone}",
    anchors = "${anchors}",
    loss_objective = "${loss_objective}",
    num_classes = 8 + 1, # add one for background
    anchor_prob_initialization = False
)