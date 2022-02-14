import os
import sys
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from helper_net import Ops

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)  # model
sys.path.append(os.path.join(ROOT_DIR, 'Pointnet2.PyTorch'))

from pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule

negative_slope = 0.2


# 1.backbone
class backbone_pointnet2(nn.Module):
    def __init__(self, is_train):
        super(backbone_pointnet2, self).__init__()
        self.sa1 = PointnetSAModule(mlp=[6, 32, 32, 64], npoint=1024, radius=0.1, nsample=32, bn=True)
        self.sa2 = PointnetSAModule(mlp=[64, 64, 64, 128], npoint=256, radius=0.2, nsample=64, bn=True)
        self.sa3 = PointnetSAModule(mlp=[128, 128, 128, 256], npoint=64, radius=0.4, nsample=128, bn=True)
        self.sa4 = PointnetSAModule(mlp=[256, 256, 256, 512], npoint=None, radius=None, nsample=None, bn=True)
        self.fp4 = PointnetFPModule(mlp=[768, 256, 256])
        self.fp3 = PointnetFPModule(mlp=[384, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[320, 256, 128])
        self.fp1 = PointnetFPModule(mlp=[137, 128, 128, 128, 128])
        self.is_train = is_train
        self.conv1 = nn.Conv2d(128, 128, kernel_size=(1, 1))
        self.lrelu1 = nn.LeakyReLU(negative_slope)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        self.lrelu2 = nn.LeakyReLU(negative_slope)
        self.drop = nn.Dropout()
        self.conv3 = nn.Conv2d(64, 13, kernel_size=(1, 1), stride=(1, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, xyz, points):
        points_num = xyz.size()[1]
        l1_xyz, l1_points = self.sa1(xyz.contiguous(), points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz.contiguous(), l1_xyz, torch.cat((xyz.transpose(1, 2), points), dim=1), l1_points)
        global_features = l4_points.view(-1, 512)
        l0_points = l0_points.transpose(1, 2)
        point_features = l0_points

        # sem
        l0_points = l0_points.transpose(1, 2)
        l0_points = l0_points[:, :, None, :]
        sem1 = self.lrelu1(self.conv1(l0_points))
        sem2 = self.lrelu2(self.conv2(sem1))
        sem2 = self.drop(sem2)
        sem3 = self.conv3(sem2)
        sem4 = torch.reshape(sem3.transpose(1, 2), [-1, points_num, 13])
        y_sem_pred = self.softmax(sem4)

        return point_features, global_features, y_sem_pred, sem4


# class backbone_sem(nn.Module):
#     def __init__(self):
#         super(backbone_sem, self).__init__()
#         self.conv1 = nn.Conv2d(128, 128, kernel_size=(1, 1))
#         self.lrelu1 = nn.LeakyReLU(negative_slope)
#         self.conv2 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
#         self.lrelu2 = nn.LeakyReLU(negative_slope)
#         self.drop = nn.Dropout()
#         self.conv3 = nn.Conv2d(64, 13, kernel_size=(1, 1), stride=(1, 1))
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, xyz, l0_points):
#         # sem
#         points_num = xyz.size()[1]
#         l0_points = l0_points.transpose(1, 2)
#         l0_points = l0_points[:, :, None, :]
#         sem1 = self.lrelu1(self.conv1(l0_points))
#         sem2 = self.lrelu2(self.conv2(sem1))
#         sem2 = self.drop(sem2)
#         sem3 = self.conv3(sem2)
#         sem4 = torch.reshape(sem3.transpose(1,2), [-1, points_num, 13])
#         y_sem_pred = self.softmax(sem4)
#
#         return y_sem_pred, sem4


# 2. bbox
class bbox_net(nn.Module):
    def __init__(self):
        super(bbox_net, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc21 = nn.Linear(512, 256)
        self.fc22 = nn.Linear(256, 256)
        # TODO 24*2*3
        self.fc3 = nn.Linear(256, 24 * 2 * 3)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 24)
        self.sigmoid = nn.Sigmoid()

    def forward(self, global_features):
        # p_num = point_features.shape[1]
        # global_features = global_features.unsqueeze(1)
        # global_features = global_features.repeat(1,p_num,1)
        # all_feature = torch.cat((global_features ,point_features) , dim = -1)

        b1 = F.leaky_relu(self.fc1(global_features), negative_slope=negative_slope)
        b2 = F.leaky_relu(self.fc21(b1), negative_slope=negative_slope)

        # sub_branch 1
        b3 = F.leaky_relu(self.fc22(b2), negative_slope=negative_slope)
        # TODO how to define bbver?
        # bbvert = F.linear(self.fc3(b3), torch.randn(24 * 2 * 3, 24 * 2 * 3))
        bbvert = torch.reshape(self.fc3(b3), [-1, 24, 2, 3])
        points_min = torch.min(bbvert, dim=-2).values[:, :, None, :]
        points_max = torch.max(bbvert, dim=-2).values[:, :, None, :]
        # bb_center = self.sigmoid(self.fc3_2(b3))
        y_bbvert_pred = torch.cat([points_min, points_max], dim=-2)

        # sub_branch 2
        b4 = F.leaky_relu(self.fc4(b2), negative_slope=negative_slope)
        # TODO out_dim????? y_bbscore_pred = tf.sigmoid(Ops.fc(b4, out_d=self.bb_num * 1, name='y_bbscore_pred'))
        y_bbscore_pred = self.sigmoid(self.fc5(b4))

        return y_bbvert_pred, y_bbscore_pred


# 3. pmask
class pmask_net(nn.Module):
    def __init__(self, p_f_num):
        super(pmask_net, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.lrelu1 = nn.LeakyReLU(negative_slope)
        self.conv1 = nn.Conv2d(1, 256, (1, p_f_num))
        self.lrelu2 = nn.LeakyReLU(negative_slope)
        self.conv2 = nn.Conv2d(512, 128, (1, 1))
        self.lrelu3 = nn.LeakyReLU(negative_slope)
        self.conv3 = nn.Conv2d(128, 128, (1, 1))
        self.lrelu4 = nn.LeakyReLU(negative_slope)
        self.conv4 = nn.Conv2d(1, 64, (1, 135))
        self.lrelu5 = nn.LeakyReLU(negative_slope)
        self.conv5 = nn.Conv2d(64, 32, (1, 1))
        self.lrelu6 = nn.LeakyReLU(negative_slope)
        self.conv6 = nn.Conv2d(32, 1, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, point_features, global_features, bbox, bboxscore):
        p_num = point_features.shape[1]
        num_box = bbox.shape[1]
        global_features = self.lrelu1(self.fc1(global_features))[:, None, None, :].repeat(1, p_num, 1, 1)
        point_features = self.lrelu2(self.conv1(point_features.unsqueeze(-1).permute(0, 3, 1, 2)))
        point_features = torch.cat((point_features, global_features.permute(0, 3, 1, 2)), dim=1)
        point_features = self.lrelu3(self.conv2(point_features))
        point_features = self.lrelu4(self.conv3(point_features))
        point_features = point_features.squeeze(-1)

        # TODO add bboxscore, TF code is as follows
        bbox_info = Ops.repeat(
            torch.cat([torch.reshape(bbox, [-1, num_box, 6]), bboxscore[:, :, None]], dim=-1)[:, :, None, :],
            [1, 1, p_num, 1])
        pmask0 = point_features.transpose(1, 2).unsqueeze(1).repeat(1, num_box, 1, 1)
        pmask0 = torch.cat((pmask0, bbox_info), dim=-1)
        pmask0 = pmask0.reshape(-1, p_num, pmask0.shape[-1], 1)

        pmask1 = self.lrelu5(self.conv4(pmask0.permute(0, 3, 1, 2)))
        pmask2 = self.lrelu6(self.conv5(pmask1))
        pmask3 = self.conv6(pmask2).permute(0, 2, 3, 1)
        pmask3 = pmask3.reshape(-1, num_box, p_num)

        pred_mask = self.sigmoid(pmask3)

        return pred_mask


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=False, weight=1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.weight = weight

    def forward(self, inputs, targets):
        F_loss = -(targets >= 0.4).float() * self.alpha * ((1. - inputs) ** self.gamma) * torch.log(inputs + 1e-8) \
                 - (1. - (targets >= 0.4).float()) * (1. - self.alpha) * (inputs ** self.gamma) * torch.log(
            1. - inputs + 1e-8)

        if self.reduce:
            return F_loss * 60
        else:
            return F_loss * 60


class FocalLoss2(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=False):
        super(FocalLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, X_pc, inputs, targets):
        points_num = X_pc.size()[1]
        # valid ins
        Y_pmask_helper = targets.sum(dim=-1)
        Y_pmask_helper = Ops.cast(torch.gt(Y_pmask_helper, 0.), torch.float32)
        Y_pmask_helper = Y_pmask_helper[:, :, None].repeat([1, 1, points_num])

        targets = targets * Y_pmask_helper
        inputs = inputs * Y_pmask_helper

        F_loss = -targets * self.alpha * ((1. - inputs) ** self.gamma) * torch.log(inputs + 1e-8) \
                 - (1. - targets) * (1. - self.alpha) * (inputs ** self.gamma) * torch.log(1. - inputs + 1e-8)

        pmask_loss_focal = torch.sum(F_loss * Y_pmask_helper) / torch.sum(Y_pmask_helper)
        pmask_loss = 30 * pmask_loss_focal

        return pmask_loss


class Hungarian(nn.Module):
    def __init__(self):
        super(Hungarian, self).__init__()

    def forward(self, cost, gt_boxes):
        box_mask = np.array([[0, 0, 0], [0, 0, 0]])

        # return ordering : batch_size x num_instances
        loss_total = 0.
        batch_size, num_instances = cost.shape[:2]
        ordering = np.zeros(shape=[batch_size, num_instances]).astype(np.int32)
        for idx in range(batch_size):
            ins_gt_boxes = gt_boxes[idx]
            ins_count = 0
            for box in ins_gt_boxes:
                if np.array_equal(box, box_mask):
                    break
                else:
                    ins_count += 1
            valid_cost = cost[idx][:ins_count]
            row_ind, col_ind = linear_sum_assignment(valid_cost.cpu().detach().numpy())
            unmapped = num_instances - ins_count
            if unmapped > 0:
                rest = np.array(range(ins_count, num_instances))
                row_ind = np.concatenate([row_ind, rest])
                unmapped_ind = np.array(list(set(range(num_instances)) - set(col_ind)))
                col_ind = np.concatenate([col_ind, unmapped_ind])

            loss_total += cost[idx][row_ind, col_ind].sum()
            ordering[idx] = np.reshape(col_ind, [1, -1])
        return torch.from_numpy(ordering).cuda(), (loss_total / float(batch_size * num_instances))


class PsemCeLoss(nn.Module):
    def __init__(self):
        super(PsemCeLoss, self).__init__()

    def forward(self, inputs, targets):
        loss = nn.CrossEntropyLoss(reduction='mean')
        return loss(inputs, torch.argmax(targets, dim=1))


class BbVertLoss(nn.Module):
    def __init__(self, label):
        super(BbVertLoss, self).__init__()
        self.label = label

    def forward(self, X_pc, y_bbvert_pred, Y_bbvert):
        label = self.label
        points_num = X_pc.size()[1]
        bb_num = int(Y_bbvert.shape[1])
        points_xyz = X_pc[:, :, 0:3]
        points_xyz = Ops.repeat(points_xyz[:, None, :, :], [1, bb_num, 1, 1])

        ##### get points hard mask in each gt bbox
        gt_bbox_min_xyz = Y_bbvert[:, :, 0, :]
        gt_bbox_max_xyz = Y_bbvert[:, :, 1, :]
        gt_bbox_min_xyz = Ops.repeat(gt_bbox_min_xyz[:, :, None, :], [1, 1, points_num, 1])
        gt_bbox_max_xyz = Ops.repeat(gt_bbox_max_xyz[:, :, None, :], [1, 1, points_num, 1])
        tp1_gt = gt_bbox_min_xyz - points_xyz
        tp2_gt = points_xyz - gt_bbox_max_xyz
        tp_gt = tp1_gt * tp2_gt
        points_in_gt_bbox_prob = Ops.cast(
            torch.eq(torch.mean(Ops.cast(torch.ge(tp_gt, 0.), torch.float32), dim=-1),
                     torch.tensor([1.0], device=torch.device('cuda'))),
            torch.float32)

        ##### get points soft mask in each pred bbox
        pred_bbox_min_xyz = y_bbvert_pred[:, :, 0, :]
        pred_bbox_max_xyz = y_bbvert_pred[:, :, 1, :]
        pred_bbox_min_xyz = Ops.repeat(pred_bbox_min_xyz[:, :, None, :], [1, 1, points_num, 1])
        pred_bbox_max_xyz = Ops.repeat(pred_bbox_max_xyz[:, :, None, :], [1, 1, points_num, 1])
        tp1_pred = pred_bbox_min_xyz - points_xyz
        tp2_pred = points_xyz - pred_bbox_max_xyz
        tp_pred = 100 * tp1_pred * tp2_pred
        tp_pred = torch.max(torch.min(tp_pred, torch.empty(tp_pred.shape, device=torch.device('cuda')).fill_(20.)),
                            torch.empty(tp_pred.shape, device=torch.device('cuda')).fill_(-20.0))
        # tp_pred = torch.maximum(torch.minimum(tp_pred, torch.tensor(20.0)), torch.tensor(-20.0))
        points_in_pred_bbox_prob = 1.0 / (1.0 + torch.exp(-1.0 * tp_pred))
        points_in_pred_bbox_prob = torch.min(points_in_pred_bbox_prob, dim=-1).values

        ##### helper -> the valid bbox (the gt boxes are zero-padded during data processing, pickup valid ones here)
        Y_bbox_helper = torch.sum(torch.reshape(Y_bbvert, [-1, bb_num, 6]), dim=-1)
        Y_bbox_helper = Ops.cast(torch.gt(Y_bbox_helper, 0.), torch.float32)

        ##### 1. get ce loss of valid/positive bboxes, don't count the ce_loss of invalid/negative bboxes
        Y_bbox_helper_tp1 = Ops.repeat(Y_bbox_helper[:, :, None], [1, 1, points_num])
        bbox_loss_ce_all = -points_in_gt_bbox_prob * torch.log(points_in_pred_bbox_prob + (1e-8)) \
                           - (1. - points_in_gt_bbox_prob) * torch.log(1. - points_in_pred_bbox_prob + 1e-8)
        bbox_loss_ce_pos = torch.sum(bbox_loss_ce_all * Y_bbox_helper_tp1) / torch.sum(Y_bbox_helper_tp1)
        bbox_loss_ce = bbox_loss_ce_pos

        ##### 2. get iou loss of valid/positive bboxes
        TP = torch.sum(points_in_pred_bbox_prob * points_in_gt_bbox_prob, dim=-1)
        FP = torch.sum(points_in_pred_bbox_prob, dim=-1) - TP
        FN = torch.sum(points_in_gt_bbox_prob, dim=-1) - TP
        bbox_loss_iou_all = TP / (TP + FP + FN + 1e-6)
        bbox_loss_iou_all = -1.0 * bbox_loss_iou_all
        bbox_loss_iou_pos = torch.sum(bbox_loss_iou_all * Y_bbox_helper) / torch.sum(Y_bbox_helper)
        bbox_loss_iou = bbox_loss_iou_pos

        ##### 3. get l2 loss of both valid/positive bboxes
        bbox_loss_l2_all = (Y_bbvert - y_bbvert_pred) ** 2
        bbox_loss_l2_all = torch.mean(torch.reshape(bbox_loss_l2_all, [-1, bb_num, 6]), dim=-1)
        bbox_loss_l2_pos = torch.sum(bbox_loss_l2_all * Y_bbox_helper) / torch.sum(Y_bbox_helper)

        ## to minimize the 3D volumn of invalid/negative bboxes, it serves as a regularizer to penalize false pred bboxes
        ## it turns out to be quite helpful, but not discussed in the paper
        bbox_pred_neg = Ops.repeat((1. - Y_bbox_helper)[:, :, None, None], [1, 1, 2, 3]) * y_bbvert_pred
        bbox_loss_l2_neg = (bbox_pred_neg[:, :, 0, :] - bbox_pred_neg[:, :, 1, :]) ** 2
        bbox_loss_l2_neg = torch.sum(bbox_loss_l2_neg) / (torch.sum(1. - Y_bbox_helper) + 1e-8)

        bbox_loss_l2 = bbox_loss_l2_pos + bbox_loss_l2_neg

        #####
        if label == 'use_all_ce_l2_iou':
            bbox_loss = bbox_loss_ce + bbox_loss_l2 + bbox_loss_iou
        elif label == 'use_both_ce_l2':
            bbox_loss = bbox_loss_ce + bbox_loss_l2
        elif label == 'use_both_ce_iou':
            bbox_loss = bbox_loss_ce + bbox_loss_iou
        elif label == 'use_both_l2_iou':
            bbox_loss = bbox_loss_l2 + bbox_loss_iou
        elif label == 'use_only_ce':
            bbox_loss = bbox_loss_ce
        elif label == 'use_only_l2':
            bbox_loss = bbox_loss_l2
        elif label == 'use_only_iou':
            bbox_loss = bbox_loss_iou
        else:
            bbox_loss = None
            print('bbox loss label error!')
            exit()

        return bbox_loss, bbox_loss_l2, bbox_loss_ce, bbox_loss_iou


class BbScoreLoss(nn.Module):
    def __init__(self):
        super(BbScoreLoss, self).__init__()

    def forward(self, y_bbscore_pred, Y_bbvert):
        bb_num = int(Y_bbvert.shape[1])

        ##### helper -> the valid bbox
        Y_bbox_helper = torch.sum(torch.reshape(Y_bbvert, [-1, bb_num, 6]), dim=-1)
        Y_bbox_helper = Ops.cast(torch.gt(Y_bbox_helper, 0.), torch.float32)

        ##### bbox score loss
        bbox_loss_score = torch.mean(-Y_bbox_helper * torch.log(y_bbscore_pred + 1e-8)
                                     - (1. - Y_bbox_helper) * torch.log(1. - y_bbscore_pred + 1e-8))
        return bbox_loss_score
