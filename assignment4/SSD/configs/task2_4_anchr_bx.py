from tops.config import LazyCall as L
from ssd.modeling import AnchorBoxes


from .task2_4 import (
    train,
    optimizer,
    anchors,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    # anchors,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map
)


anchors = L(AnchorBoxes)(
    feature_sizes= [[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    strides= [[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes= [[8, 8], [16, 16], [32, 32], [56, 56], [86, 86], [128, 128], [128, 400]],
    aspect_ratios= [[2, 4], [2, 4], [2, 4], [2, 3], [2, 3], [2, 3]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)