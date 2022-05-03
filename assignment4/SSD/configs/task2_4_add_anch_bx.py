from tops.config import LazyCall as L
from ssd.modeling import AnchorBoxes
from ssd.modeling.retina_net_w_init import RetNetWInit
from ssd.modeling.backbones import FPN
from ssd.modeling.backbones import FPN_mod

# from .task2_4 import (
from .task2_3_w_init import (
    train,
    optimizer,
    anchors,
    schedulers,
    loss_objective,
    # model,
    # backbone,
    data_train,
    data_val,
    # anchors,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map
)

# train.batch_size=16

anchors = L(AnchorBoxes)(
    feature_sizes= [[64, 512], [32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    strides= [[2, 2], [4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes= [[8, 8], [16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
    aspect_ratios= [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

backbone = L(FPN_mod)(
    input_channels=[256, 512, 1024, 2048, 256, 128, 64],
    output_channels=[256, 256, 256, 256, 256, 256, 256],
    #output_channels=[128, 256, 128, 512, 64, 64],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)

model = L(RetNetWInit)(
    feature_extractor = "${backbone}",
    anchors = "${anchors}",
    loss_objective = "${loss_objective}",
    num_classes = 8 + 1, # add one for background
    anchor_prob_initialization = False
)