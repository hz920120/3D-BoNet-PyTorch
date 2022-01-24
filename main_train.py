import os
import torch
from torch import optim

from BoNetMLP import *

gpu_num = 0


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
    ]
    return devices if devices else [torch.device('cpu')]


# x = torch.ones(2, 3, device=try_gpu())
# print(x)

# def train(net, data):
#     for epoch in range(0, 51, 1):
#         l_rate = max(0.0005/(2**(epoch//20)), 0.00001)
#
#         data.shuffle_train_files()
#         total_train_batch_num = data.total_train_batch_num
#         print('total train batch num:', total_train_batch_num)
#
#
#         for i in range(total_train_batch_num):
#             # training
#             data.

def Get_instance_size(batch_pc, batch_group, bat_bbvert, instance_size):
    batch_size = batch_pc.shape[0]
    num_point = batch_pc.shape[1]
    gt_instance_size = torch.zeros((batch_size, 20)).cuda()
    for i in range(batch_size):
        pc = batch_pc[i]
        pc_group = batch_group[i]
        pc_sem = batch_sem[i]
        pc_group_unique = torch.unique(pc_group)
        pc_bbvert = bat_bbvert[i]
        idx = -1
        for ins in pc_group_unique:
            if ins == -1: continue
            idx += 1
            pos = (pc_group == ins).nonzero().squeeze(-1)
            i_size = (pc_bbvert[ins.long(), 1, :] - pc_bbvert[ins.long(), 0, :]) / 2
            size_cal = torch.sum(torch.abs(instance_size - i_size), dim=1)
            size_idx = torch.argmin(size_cal)
            gt_instance_size[i, ins.long()] = size_idx

    return gt_instance_size


def Get_instance_center(batch_pc, batch_group, batch_bbvert):
    batch_size = batch_pc.shape[0]
    num_point = batch_pc.shape[1]
    gt_mask = torch.zeros((batch_size, num_point)).cuda()
    sd1 = 0.004
    sd2 = 0.012
    for i in range(batch_size):
        pc = batch_pc[i]
        pc_group = batch_group[i]
        pc_group_unique = torch.unique(pc_group)
        pc_bbvert = batch_bbvert[i]
        pc_bbvert = pc_bbvert[:, 1, :] - pc_bbvert[:, 0, :]
        pc_bbvert = pc_bbvert[:, 0] * pc_bbvert[:, 1] * pc_bbvert[:, 2]
        count = 0

        for ins in pc_group_unique:
            sd = sd1 + pc_bbvert[count] * (sd2 - sd1)
            pos = (pc_group == ins).nonzero().squeeze(-1)
            pc_instance = pc[(pc_group == ins), :]
            center = torch.mean((pc_instance), dim=0)[:3].unsqueeze(0)
            dist = torch.sum((pc_instance[:, 0:3] - center) ** 2, dim=1)
            new_idx = torch.topk(dist, 1, largest=False)[1]
            new_center = pc_instance[new_idx, :3]
            new_dist = torch.sum((pc_instance[:, 0:3] - new_center) ** 2, dim=1)
            final_value = torch.exp(-(new_dist / (2 * sd))) / (torch.sqrt(sd) * np.sqrt(2 * math.pi))
            final_min = torch.min(final_value)
            final_max = torch.max(final_value)
            if final_max == final_min:
                final_value = final_value
            else:
                final_value = (final_value - final_min) / (final_max - final_min)
            count = count + 1

            gt_mask[i, pos] = final_value.cuda()

    gt_mask = torch.clamp(gt_mask, min=0.0, max=1.0)

    return gt_mask


if __name__ == '__main__':

    # from main_3D_BoNet import BoNet
    # from helper_data_s3dis import Data_Configs as Data_Configs
    #
    # configs = Data_Configs()
    # net = BoNetMLP(configs = configs)
    # # net.creat_folders(name='log', re_train=False)
    # net.build_net()
    #
    # ####
    from helper_data_s3dis import Data_S3DIS as Data

    train_areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
    test_areas = ['Area_5']
    #
    dataset_path = './data_s3dis/'
    data = Data(dataset_path, train_areas, test_areas, train_batch_size=4)
    # train(net, data)

    # some parameters
    batch_size = 4
    num_feature = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch = 0
    l_rate = max(0.0005 / (2 ** (epoch // 20)), 0.00001)
    instance_sizes = [[0.48698184 ,0.25347686 , 0.42515151] , [0.26272924 ,0.25347686, 0.42515151] , [0.26272924 , 0.48322966 ,0.42515151] , [0.07527845,0.25347686 ,0.42515151] , [0.26272924, 0.06318584, 0.42515151],[0.48698184 , 0.06318584, 0.13261693]]


    train_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    instance_size = torch.zeros((6,3))
    instance_sizes = np.array(instance_sizes)
    instance_sizes = torch.tensor(instance_sizes)
    
    for i in range(6):
        instance_size[i] = instance_sizes[i]

    if torch.cuda.is_available():
        instance_size = instance_size.cuda()

    # backbone_pointnet2
    backbone = backbone_pointnet2(is_train=True)
    backbone = backbone.to(device)

    # bbox_net
    bbox_net = bbox_net()
    bbox_net = bbox_net.to(device)

    # pmask_net
    pmask_net = pmask_net(num_feature)
    pmask_net = pmask_net.to(device)

    optim_params = [
        {'params': backbone.parameters(), 'lr': l_rate, 'betas': (0.9, 0.999), 'eps': 1e-08},
        {'params': bbox_net.parameters(), 'lr': l_rate, 'betas': (0.9, 0.999), 'eps': 1e-08},
        {'params': pmask_net.parameters(), 'lr': l_rate, 'betas': (0.9, 0.999), 'eps': 1e-08},
    ]
    optimizer = optim.Adam(optim_params)
    for ep in range(0, 51, 1):
        l_rate = max(0.0005 / (2 ** (ep // 20)), 0.00001)
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            bat_pc, _, _, bat_psem_onehot, bat_bbvert, bat_pmask = data
            if torch.cuda.is_available():
                bat_pc, bat_ins, batch_sem, bat_bbvert, bat_pmask = bat_pc.cuda(), bat_ins.cuda(), batch_sem.cuda(), bat_bbvert.cuda(), bat_pmask.cuda()
            point_features, global_features = backbone(bat_pc[:, :, 0:3], bat_pc[:, :, 3:9].transpose(1, 2))

            y_bbvert_pred_raw, y_bbscore_pred_raw = bbox_net(global_features)
            # TODO Ops.bbvert_association & Ops.bbscore_association
            y_bbvert_pred, pred_bborder = Ops.bbvert_association(bat_pc, y_bbvert_pred_raw, bat_bbvert,
                                                                 label='use_all_ce_l2_iou')
            y_bbscore_pred = Ops.bbscore_association(y_bbscore_pred_raw, pred_bborder)

            # loss bbox
            bbvert_loss, bbvert_loss_l2, bbvert_loss_ce, bbvert_loss_iou = \
                Ops.get_loss_bbvert(bat_pc, y_bbvert_pred, bat_bbvert, label='use_all_ce_l2_iou')
            bbscore_loss = Ops.get_loss_bbscore(y_bbscore_pred, bat_bbvert)
            sum_bbox_vert_loss = tf.summary.scalar('bbvert_loss', bbvert_loss)
            sum_bbox_vert_loss_l2 = tf.summary.scalar('bbvert_loss_l2', bbvert_loss_l2)
            sum_bbox_vert_loss_ce = tf.summary.scalar('bbvert_loss_ce', bbvert_loss_ce)
            sum_bbox_vert_loss_iou = tf.summary.scalar('bbvert_loss_iou', bbvert_loss_iou)
            sum_bbox_score_loss = tf.summary.scalar('bbscore_loss', bbscore_loss)

            pred_mask_pred = pmask_net(point_features, global_features, y_bbvert_pred_raw, y_bbscore_pred)

            # get gt instance size 
            gt_instance_size = Get_instance_size(bat_pc, bat_ins, bat_bbvert, instance_size)

            # pick topk heatmap value
            gt_instance_idx = torch.topk(gt_center, max_output_size, dim=1)[1]
            c_gt_instance_idx = (gt_instance_idx >= 0).nonzero()
            gt_instance_idx = torch.cat((c_gt_instance_idx[:, 0].unsqueeze(-1), gt_instance_idx.view(-1, 1)), 1)
            gt_instance_idx = gt_instance_idx.detach()
            group_label_seed = bat_ins[gt_instance_idx[:, 0], gt_instance_idx[:, 1]].view(-1, max_output_size)
            idx = (group_label_seed >= 0).nonzero()
            group_label_seed_aug = torch.cat((idx[:, 0].unsqueeze(-1), group_label_seed.long().view(-1, 1)), 1)
            pc_ins_bound_gt = bat_bbvert[group_label_seed_aug[:, 0], group_label_seed_aug[:, 1], :, :].view(-1,
                                                                                                            max_output_size,
                                                                                                            2,
                                                                                                            3).float()

            # get instance size, bbox and mask
            size_gt = gt_instance_size[group_label_seed_aug[:, 0], group_label_seed_aug[:, 1]].view(-1,
                                                                                                    max_output_size).float()
            bbox_shape = bat_bbvert[:, :, 1, :] - bat_bbvert[:, :, 0, :]
            bbox_center = torch.mean((bat_bbvert), dim=-2)
            bbox_gt = torch.cat((bbox_center, bbox_shape), dim=-1)
            bbox_gt = bbox_gt[group_label_seed_aug[:, 0], group_label_seed_aug[:, 1], :].view(-1, max_output_size, 6)
            mask_gt = bat_pmask[group_label_seed_aug[:, 0], group_label_seed_aug[:, 1], :].view(-1, max_output_size,
                                                                                                4096)

            # loss pred_mask
            pmask_loss = Ops.get_loss_pmask(bat_pc, pred_mask_pred, mask_gt)


            end_2_end_loss = bbvert_loss + bbscore_loss  + pmask_loss + psemce_loss
            end_2_end_loss.backward()
            optimizer.step()