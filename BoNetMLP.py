import os
import sys
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

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

        l0_points = l0_points[:, :, None, :]
        sem1 = F.leaky_relu(F.conv2d(l0_points, torch.rand(4096, 4096, 1, 1).cuda()),
                            negative_slope)
        # sem2 = Ops.xxlu(Ops.conv2d(sem1, k=(1, 1), out_c=64, str=1, pad='VALID', name='sem2'), label='lrelu')
        sem2 = F.leaky_relu(F.conv2d(sem1, torch.rand(4096, 4096, 1, 1).cuda(), stride=2),
                            negative_slope)
        # sem2 = Ops.dropout(sem2, keep_prob=0.5, is_train=self.is_train, name='sem2_dropout')
        sem2 = F.dropout(sem2, p=0.5, training=self.is_train)
        # sem3 = Ops.conv2d(sem2, k=(1, 1), out_c=self.sem_num, str=1, pad='VALID', name='sem3')
        # TODO self.sem_num = 13?
        sem3 = F.conv2d(sem2, torch.rand(4096, 4096, 1, 4).cuda(), padding=2, stride=5)
        # TODO points_num ?
        # sem3 = torch.reshape(sem3, [-1, points_num, self.sem_num])
        sem4 = torch.reshape(sem3, [-1, points_num])
        y_sem_pred = F.softmax(sem4)

        return point_features, global_features, y_sem_pred, sem4


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
        # self.bn1= nn.BatchNorm1d(512)
        # self.bn2= nn.BatchNorm1d(256)
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
        points_min = torch.min(bbvert, dim=-2).values[:,:,None,:]
        points_max = torch.max(bbvert, dim=-2).values[:,:,None,:]
        # bb_center = self.sigmoid(self.fc3_2(b3))
        y_bbvert_pred = torch.cat([points_min, points_max], dim=-2)

        # sub_branch 2
        b4 = F.leaky_relu(self.fc4(b2), negative_slope=negative_slope)
        # TODO out_dim????? y_bbscore_pred = tf.sigmoid(Ops.fc(b4, out_d=self.bb_num * 1, name='y_bbscore_pred'))
        y_bbscore_pred = torch.sigmoid(self.fc5(b4))

        return y_bbvert_pred, y_bbscore_pred


# 3. pmask
class pmask_net(nn.Module):
    def __init__(self, p_f_num):
        super(pmask_net, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.conv1 = nn.Conv2d(1, 256, (1, p_f_num))
        self.conv2 = nn.Conv2d(512, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))
        self.conv4 = nn.Conv2d(1, 64, (1, 134))
        self.conv5 = nn.Conv2d(64, 32, (1, 1))
        self.conv6 = nn.Conv2d(32, 1, (1, 1))
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.sigmoid = nn.Sigmoid()

    def forward(self, point_features, global_features, bbox, bboxscore):
        p_num = point_features.shape[1]
        num_box = bbox.shape[1]
        global_features = F.leaky_relu(self.fc1(global_features), negative_slope=negative_slope).unsqueeze(1).unsqueeze(
            1).repeat(1, p_num, 1, 1)
        point_features = F.leaky_relu(self.conv1(point_features.unsqueeze(-1).permute(0, 3, 1, 2)), negative_slope=0.2)
        point_features = torch.cat((point_features, global_features.permute(0, 3, 1, 2)), dim=1)
        point_features = F.leaky_relu(self.conv2(point_features), negative_slope=negative_slope)
        point_features = F.leaky_relu(self.conv3(point_features), negative_slope=negative_slope)
        point_features = point_features.squeeze(-1)

        # TODO add bboxscore, TF code is as follows
        bbox_info = torch.tile(torch.cat([torch.reshape(bbox, [-1, p_num, 6]), bboxscore[:,:,None]],dim=-1)[:,:,None,:], [1,1,p_num,1])
        pmask0 = point_features.transpose(1, 2).unsqueeze(1).repeat(1, num_box, 1, 1)
        pmask0 = torch.cat((pmask0, bbox_info), dim=-1)
        pmask0 = pmask0.view(-1, p_num, pmask0.shape[-1], 1)

        pmask1 = F.leaky_relu(self.conv4(pmask0.permute(0, 3, 1, 2)), negative_slope=negative_slope)
        pmask2 = F.leaky_relu(self.conv5(pmask1), negative_slope=negative_slope)
        pmask3 = self.conv6(pmask2).permute(0, 2, 3, 1)
        pmask3 = pmask3.view(-1, num_box, p_num)

        pred_mask = self.sigmoid(pmask3)

        return pred_mask

