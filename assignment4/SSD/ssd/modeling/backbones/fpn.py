import torch
from torch import nn
import torchvision.models as tvm


class FPN(torch.nn.Module):
    """
    This is a resnet backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]], 
            type: str, 
            pretrained: bool):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        self.type = type
        self.pretrained = pretrained

        self.resnet = tvm.resnet34(pretrained=self.pretrained)
        modules = list(self.resnet.children())[:-2]
        backbone = nn.Sequential(*modules)

        self.feature_extractor = backbone
        
        '''
        # backbone
        if type == 'resnet_18':
            self.resnet = torchvision.models.resnet18(pretrained=self.pretrained)
            modules = list(self.resnet.children())[:-1]
            backbone = nn.Sequential(*modules)
            backbone.out_channels = 512
        elif type == 'resnet_34':
            self.resnet = torchvision.models.resnet34(pretrained=self.pretrained)
            modules = list(self.resnet.children())[:-1]
            backbone = nn.Sequential(*modules)
            backbone.out_channels = 512
        elif type == 'resnet_50':
            self.resnet = torchvision.models.resnet50(pretrained=self.pretrained)
            modules = list(self.resnet.children())[:-1]
            backbone = nn.Sequential(*modules)
            backbone.out_channels = 2048
        elif type == 'resnet_101':
            self.resnet = torchvision.models.resnet101(pretrained=self.pretrained)
            modules = list(self.resnet.children())[:-1]
            backbone = nn.Sequential(*modules)
            backbone.out_channels = 2048
        elif type == 'resnet_152':
            self.resnet = torchvision.models.resnet152(pretrained=self.pretrained)
            modules = list(self.resnet.children())[:-1]
            backbone = nn.Sequential(*modules)
            backbone.out_channels = 2048
        elif type == 'resnet_50_modified_stride_1':
            self.resnet = resnet50(pretrained=self.pretrained)
            modules = list(self.resnet.children())[:-1]
            backbone = nn.Sequential(*modules)
            backbone.out_channels = 2048
    
        self.feature_extractor = self.resnet.children()
        '''

def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        for layer in self.feature_extractor:
            out_features.append(layer(x))
            x = out_features[-1]

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)


