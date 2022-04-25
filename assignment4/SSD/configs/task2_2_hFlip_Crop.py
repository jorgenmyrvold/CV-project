from tops.config import LazyCall as L
import torchvision
from ssd.data.transforms import (ToTensor, Normalize, Resize,GroundTruthBoxesToAnchors, RandomHorizontalFlip, RandomSampleCrop)
from .task2_1 import (
    train,
    optimizer,
    anchors,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    anchors,
    # train_cpu_transform,
    # val_cpu_transform,
    # gpu_transform,
    label_map
)

train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
    L(RandomHorizontalFlip)(),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])
val_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
])
gpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878]),
])
