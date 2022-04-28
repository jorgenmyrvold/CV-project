
from tops.config import LazyCall as L
from ssd.modeling.retina_net_w_init import RetNetWInit
from ssd.modeling.backbones import FPN
import torchvision
import torch
from .utils import get_dataset_dir
from ssd.data.transforms import (ToTensor, Normalize, Resize,GroundTruthBoxesToAnchors, RandomHorizontalFlip, RandomSampleCrop)
from ssd.data import TDT4265Dataset
from ssd import utils


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

# data_train['img_folder'] = get_dataset_dir("tdt4265_2022_updated"),
# img_folder=get_dataset_dir("tdt4265_2022_updated")



data_train = dict(
    dataset=L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022_updated"),
    train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
        L(RandomSampleCrop)(),
        L(ToTensor)(),
        L(Resize)(imshape="${train.imshape}"),
        L(RandomHorizontalFlip)(),
        L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
    ]),
    annotation_file=get_dataset_dir("tdt4265_2022_updated/train_annotations.json")),
    dataloader=L(torch.utils.data.DataLoader)(
        dataset="${..dataset}", num_workers=4, pin_memory=True, shuffle=True, batch_size="${...train.batch_size}", collate_fn=utils.batch_collate,
        drop_last=True
    ),
    # GPU transforms can heavily speedup data augmentations.
    gpu_transform=L(torchvision.transforms.Compose)(transforms=[
        L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])  # Normalize has to be applied after ToTensor (GPU transform is always after CPU)
    ])
)
data_val = dict(
    dataset=L(TDT4265Dataset)(
        img_folder=get_dataset_dir("tdt4265_2022_updated"),
        transform=L(torchvision.transforms.Compose)(transforms=[
            L(ToTensor)(),
            L(Resize)(imshape="${train.imshape}"),
        ]),
        annotation_file=get_dataset_dir("tdt4265_2022_updated/val_annotations.json")
    ),
    dataloader=L(torch.utils.data.DataLoader)(
        dataset="${..dataset}", num_workers=4, pin_memory=True, shuffle=False, batch_size="${...train.batch_size}", collate_fn=utils.batch_collate_val
    ),
    gpu_transform=L(torchvision.transforms.Compose)(transforms=[
        L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
    ])
)
