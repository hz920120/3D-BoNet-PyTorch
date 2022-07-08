import torch
import torch.nn as nn
import torch.nn.functional as F
import models.pytorch_lib as pt_utils
from data_process import DataProcessing as DP
import numpy as np
from sklearn.metrics import confusion_matrix


class RandLA(nn.Module):

    def __init__(self, num_layers, d_out, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.d_out = d_out
        self.class_weights = DP.get_class_weights('S3DIS')

        self.fc0 = pt_utils.Conv1d(6, 8, kernel_size=1, bn=True)
        # self.fc0 = pt_utils.Conv1d(12, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.num_layers):
            d_out = self.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        self.global_conv1 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

        self.decoder_blocks = nn.ModuleList()
        for j in range(self.num_layers):
            if j < self.num_layers - 1:
                d_in = d_out + 2 * self.d_out[-j - 2]
                d_out = 2 * self.d_out[-j - 2]
            else:
                d_in = 4 * self.d_out[-self.num_layers]
                d_out = 2 * self.d_out[-self.num_layers]
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))

        self.point_conv1 = pt_utils.Conv2d(d_out, 128, kernel_size=(1, 1), bn=True)

        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1, 1), bn=True)
        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(32, self.num_classes, kernel_size=(1, 1), bn=False, activation=None)

    def forward(self, end_points, device):

        features = end_points['features'][:, 0:6].to(device)  # B,12,N
        # features = end_points['features'].to(device)  # B,12,N
        features = self.fc0(features)  # B,8,N

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i].to(device),
                                                     end_points['neigh_idx'][i].to(device))

            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i].to(device))
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)

        # ###########################Encoder############################

        global_features = self.global_conv1(features)
        global_features = F.adaptive_avg_pool2d(global_features, output_size=(1, 1))
        # global_features = F.avg_pool2d(global_features, kernel_size=(80,1))
        global_features = torch.squeeze(global_features, dim=-1)
        global_features = torch.squeeze(global_features, dim=-1)

        features = self.decoder_0(f_encoder_list[-1])  # B,C=1024,N,1

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1].to(device))
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))

            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        point_features = torch.squeeze(self.point_conv1(f_decoder_list[-1]),
                                       dim=-1)  # (B,C=128,N). f_decoder_list[-1] B,C=32,N,1

        ### sem
        features = self.fc1(features)  # B,C=64,N,1
        features = self.fc2(features)  # B,C=32,N,1
        features = self.dropout(features)
        features = self.fc3(features)  # B,C=40,N,1
        y_psem_logits = features.squeeze(3)  # B,C=40,N

        end_points['logits'] = y_psem_logits  # B,40,N

        y_sem_pred = F.log_softmax(y_psem_logits, dim=1)  # B,40,N

        return point_features.transpose(1, 2), global_features, y_sem_pred.permute(0, 2, 1).contiguous(), y_psem_logits.permute(0, 2, 1).contiguous()

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features


class Psem_Loss_ignored(nn.Module):
    def __init__(self, device):
        super(Psem_Loss_ignored, self).__init__()
        self.class_weights = torch.from_numpy(DP.get_class_weights('S3DIS')).float().to(device)
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, y_psem_logits, Y_psem): # B,N,13. B,N,13
        y_psem_logits_rep = y_psem_logits.permute(0,2,1).contiguous() # B,13,N
        Y_psem_rep = torch.argmax(Y_psem, dim=-1) # B,N
        loss = self.cross_entropy(y_psem_logits_rep, Y_psem_rep) # default reduction is 'mean'
        return loss


def compute_acc(end_points):
    logits = end_points['valid_logits']
    labels = end_points['valid_labels']
    logits = logits.max(dim=1)[1]
    acc = (logits == labels).sum().float() / float(labels.shape[0])
    end_points['acc'] = acc
    return acc, end_points


class IoUCalculator:
    def __init__(self, cfg):
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg

    def add_data(self, end_points):
        logits = end_points['valid_logits']
        labels = end_points['valid_labels']
        pred = logits.max(dim=1)[1]
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        val_total_correct = 0
        val_total_seen = 0

        correct = np.sum(pred_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.cfg.num_classes, 1))
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / float(
                    self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        return mean_iou, iou_list


class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out // 2, kernel_size=(1, 1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc + shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  # d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out // 2, kernel_size=(1, 1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out // 2)

        self.mlp2 = pt_utils.Conv2d(d_out // 2, d_out // 2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz, relative_dis = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)),
                                             neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat, relative_dis)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)),
                                             neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat, relative_dis)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        relative_dis = torch.sqrt(
            torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz],
                                     dim=-1)  # batch*npoint*nsamples*10
        return relative_feature, relative_dis

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        idx = index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2])
        features = torch.gather(pc, 1, idx)
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature_set, relative_dis):  # relative_dis: B,N,K,1
        # inverse distance weights with normalization
        # B, N, K, C = relative_dis.size()
        # relative_dis_min = torch.min(relative_dis, dim=2, keepdim=True)
        # relative_dis_max = torch.max(relative_dis, dim=2, keepdim=True)
        # dd = relative_dis_max - relative_dis_min
        # relative_dis_norm = (relative_dis - relative_dis_min) / dd
        # inv_relative_dis_norm = 1 - relative_dis_norm
        # p = inv_relative_dis_norm.cpu().numpy()
        # inv_dis_scores = F.softmax(inv_relative_dis_norm, dim=2)
        # inv_dis_scores = inv_dis_scores.permute((0, 3, 1, 2))  # B,C,N,K

        # origin
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores

        # add inverse distance scores with att_scores parallel
        # att_activation = self.fc(feature_set)
        # att_scores = F.softmax(att_activation, dim=3)
        # f_agg = feature_set * att_scores * inv_dist_scores

        # reduce sum in dim 3
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


def compute_loss(end_points, cfg):
    logits = end_points['logits']
    labels = end_points['labels']

    logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)
    labels = labels.reshape(-1)

    # Boolean mask of points that should be ignored
    ignored_bool = labels == 0
    for ign_label in cfg.ignored_label_inds:
        ignored_bool = ignored_bool | (labels == ign_label)

    # Collect logits and labels that are not ignored
    valid_idx = ignored_bool == 0
    valid_logits = logits[valid_idx, :]
    valid_labels_init = labels[valid_idx]

    # Reduce label values in the range of logit shape
    reducing_list = torch.range(0, cfg.num_classes).long().cuda()
    inserted_value = torch.zeros((1,)).long().cuda()
    for ign_label in cfg.ignored_label_inds:
        reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
    loss = get_loss(valid_logits, valid_labels, cfg.class_weights)
    end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels
    end_points['loss'] = loss
    return loss, end_points


def get_loss(logits, labels, pre_cal_weights):
    # calculate the weighted cross entropy according to the inverse frequency
    class_weights = torch.from_numpy(pre_cal_weights).float().cuda()
    # one_hot_labels = F.one_hot(labels, self.config.num_classes)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    output_loss = criterion(logits, labels)
    output_loss = output_loss.mean()
    return output_loss