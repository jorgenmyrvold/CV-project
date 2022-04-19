from tops.config import LazyCall as L
import torchvision
from ssd.modeling.backbones import FPN
from .task2_2_hFlip_Crop import (
    train,
    optimizer,
    anchors,
    schedulers,
    loss_objective,
    model,
    #backbone,
    data_train,
    data_val,
    anchors,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map
)

backbone = L(FPN)(
    output_channels=[128, 256, 256, 128, 64, 64],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}")