import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F

def one_hot_encode(Y: np.ndarray, num_classes: int)->torch.Tensor:
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    temp = torch.zeros((Y.size(dim=0),num_classes))
    for ex in range(0, Y.size(dim=0)):
        temp[ex][Y[ex]] = 1
    Y = temp
    return Y


class FocalLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, anchors, num_classes, gamma, alpha):
        super().__init__()
        self.scale_xy = 1.0/anchors.scale_xy
        self.scale_wh = 1.0/anchors.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.anchors = nn.Parameter(anchors(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)
        self.gamma = gamma
        self.alpha = torch.FloatTensor(alpha).cuda()
        self.num_classes = num_classes


    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.anchors[:, :2, :])/self.anchors[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.anchors[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()
    
    def forward(self,
            bbox_delta: torch.FloatTensor, confs: torch.FloatTensor,
            gt_bbox: torch.FloatTensor, gt_labels: torch.LongTensor): # gt_labels = y
        #print("confs: ", confs.shape)
        #print("bbox_delta", bbox_delta.shape)
        """
        NA is the number of anchor boxes (by default this is 8732)
            bbox_delta: [batch_size, 4, num_anchors]
            confs: [batch_size, num_classes, num_anchors]
            gt_bbox: [batch_size, num_anchors, 4]
            gt_label = [batch_size, num_anchors]
        """
        gt_bbox = gt_bbox.transpose(1, 2).contiguous() # reshape to [batch_size, 4, num_anchors]
        
        # One-hot-encoding of gt_labels
        
        one_hot_target = F.one_hot(gt_labels, self.num_classes)
        one_hot_target = torch.transpose(one_hot_target, 1, 2)
        
        # Apply softmax to confs
        log_p_k = F.log_softmax(confs, dim=1)
        
        p_k = F.softmax(confs,dim=1)
        #calculate focal loss
        weight = torch.pow(1.0 - p_k, self.gamma)
        
        focal = weight * one_hot_target * log_p_k
        alphas = self.alpha.repeat(confs.shape[2], 1).T
        focal = -alphas.repeat(confs.shape[0],1,1) * focal
        focal_loss=torch.sum(focal)

        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask]
        gt_locations = self._loc_vec(gt_bbox)
        gt_locations = gt_locations[pos_mask]
        regression_loss = F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum")
        num_pos = gt_locations.shape[0]/4
        total_loss = regression_loss/num_pos + focal_loss/num_pos
        to_log = dict(
            regression_loss=regression_loss/num_pos,
            classification_loss=focal_loss/num_pos,
            total_loss=total_loss
        )
        return total_loss, to_log
