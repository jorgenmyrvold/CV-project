
from tops.config import LazyCall as L
import torchvision
import torch
from ssd.modeling import SSD300
from ssd.modeling.backbones import FPN
from .task2_2_hFlip_Crop import (
    train,
    optimizer,
    anchors,
    schedulers,
    loss_objective,
    #model,
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
    input_channels=[256, 512, 1024, 2048, 256, 64], 
    output_channels=[256, 256, 256, 256, 256, 256], 
    #output_channels=[128, 256, 128, 512, 64, 64],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}")

model = L(SSD300)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1  # Add 1 for background
)   
