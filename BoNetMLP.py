import os
import shutil
import sys
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from helper_net import Ops

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'Pointnet2.PyTorch'))


from pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG , PointnetSAModule
import pointnet2.pointnet2_utils as pointnet2_utils
import pointnet2.pytorch_utils as pt_utils


class backbone_pointnet2(nn.Module):
    def __init__(self, is_train):
        super(backbone_pointnet2, self).__init__()
        self.sa1 = PointnetSAModule(mlp=[6,32,32,64] , npoint=1024, radius= 0.1 , nsample= 32 , bn=True)
        self.sa2 = PointnetSAModule(mlp=[64,64,64,128] , npoint=256, radius= 0.2 , nsample= 64 , bn=True)
        self.sa3 = PointnetSAModule(mlp=[128,128,128,256] , npoint=64, radius= 0.4 , nsample= 128 , bn=True)
        self.sa4 = PointnetSAModule(mlp=[256,256,256,512] , npoint=None, radius= None , nsample= None , bn=True)
        self.fp4 = PointnetFPModule(mlp = [768,256,256])
        self.fp3 = PointnetFPModule(mlp = [384,256,256])
        self.fp2 = PointnetFPModule(mlp = [320,256,128])
        self.fp1 = PointnetFPModule(mlp = [137,128,128,128,128])
        self.is_train = is_train

    def forward(self, xyz , points):

        l1_xyz, l1_points = self.sa1(xyz.contiguous(), points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l3_points  = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points  = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points  = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points  = self.fp1(xyz.contiguous(), l1_xyz, torch.cat((xyz.transpose(1,2) ,points), dim = 1), l1_points)

        global_features = l4_points.view(-1,512)
        point_features = l0_points.transpose(1,2)

        # sem
        negative_slope=0.2
        l0_points = l0_points[:,:,None,:]
        sem1 = F.leaky_relu(F.conv2d(l0_points, torch.rand(1, 1, 128, 128), torch.rand(128), stride=1, padding=1), negative_slope)
        # sem2 = Ops.xxlu(Ops.conv2d(sem1, k=(1, 1), out_c=64, str=1, pad='VALID', name='sem2'), label='lrelu')
        sem2 = F.leaky_relu(F.conv2d(sem1, torch.rand(1, 1, 128, 64), torch.rand(64), stride=1, padding=1), negative_slope)
        # sem2 = Ops.dropout(sem2, keep_prob=0.5, is_train=self.is_train, name='sem2_dropout')
        sem2 = F.dropout(sem2, p=0.5, training=self.is_train)
        # sem3 = Ops.conv2d(sem2, k=(1, 1), out_c=self.sem_num, str=1, pad='VALID', name='sem3')
        #TODO self.sem_num = 13?
        sem3 = F.conv2d(sem2, torch.rand(1, 1, 128, 13), torch.rand(13), stride=1, padding=1)
        #TODO points_num ?
        # sem3 = torch.reshape(sem3, [-1, points_num, self.sem_num])
        sem3 = torch.reshape(sem3, [-1, self.sem_num])
        self.y_psem_logits = sem3
        y_sem_pred = F.softmax(self.y_psem_logits)

        return point_features, global_features, y_sem_pred



class BoNetMLP(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, net, GPU='0'):
        #######   1. define inputs
        # point cloud channel
        # self.X_pc = torch.empty(self.points_cc, dtype=torch.float32,  names=('X_pc',))
        #
        # # bounding box vertices -> bian kang ding dian ge shu
        # self.Y_bbvert = torch.zeros([None, self.bb_num, 2, 3], dtype=torch.float32, names=('Y_bbvert',))
        # # mask shuliang
        # self.Y_pmask = torch.zeros([None, self.bb_num, None], dtype=torch.float32, names=('Y_pmask',))
        # # point semantics -> label count
        # self.Y_psem = torch.zeros([None, None, self.sem_num], dtype=torch.float32, names='Y_psem')
        # self.is_train = torch.zeros(dtype=torch.bool, names='is_train')
        # self.lr = torch.zeros(dtype=torch.float32, names='lr')
        for block in self._modules.values():
            net = block(net)
        return net