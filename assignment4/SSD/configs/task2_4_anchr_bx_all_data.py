from tops.config import LazyCall as L
from ssd.modeling import AnchorBoxes
from .utils import get_dataset_dir

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

data_train.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
data_train.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/train_annotations.json")
data_val.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
data_val.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/val_annotations.json")


anchors = L(AnchorBoxes)(
    feature_sizes= [[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    strides= [[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes= [[8, 8], [22, 22], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
    aspect_ratios= [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)