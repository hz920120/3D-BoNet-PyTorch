import torch.utils.data
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from BoNetMLP import *
from S3DISDataLoader import S3DISDataset
from helper_net import Ops

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
    # from helper_data_s3dis import Data_S3DIS as Data
    from dataset import Data_S3DIS as Data

    writer = SummaryWriter('logs')

    train_areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
    test_areas = ['Area_5']
    #
    dataset_path = './Data_S3DIS_bak/'
    # data = S3DISDataset(split='train', data_root=dataset_path, transform=None)
    data = Data(dataset_path, train_areas, test_areas, train_batch_size=4)

    # train(net, data)

    # some parameters
    batch_size = 4
    num_feature = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    epoch = 0
    l_rate = max(0.0005 / (2 ** (epoch // 20)), 0.00001)

    # train_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    train_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
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
                bat_pc, _, _, bat_bbvert, bat_pmask, bat_psem_onehot = bat_pc.cuda(), _, _, bat_bbvert.cuda(), bat_pmask.cuda(), bat_psem_onehot.cuda()
            # point_features, global_features, y_sem_pred, y_psem_logits = backbone(bat_pc[:, :, 0:9])
            point_features, global_features, y_sem_pred, y_psem_logits = backbone(bat_pc[:, :, 0:3],
                                                                                  bat_pc[:, :, 3:9].transpose(1, 2))

            y_bbvert_pred_raw, y_bbscore_pred_raw = bbox_net(global_features)
            # # TODO Ops.bbvert_association & Ops.bbscore_association
            # y_bbvert_pred, pred_bborder = Ops.bbvert_association(bat_pc, y_bbvert_pred_raw, bat_bbvert,
            #                                                      label='use_all_ce_l2_iou')
            associate_maxtrix, Y_bbvert = Ops.bbvert_association(bat_pc, y_bbvert_pred_raw, bat_bbvert,
                                                                 label='use_all_ce_l2_iou')
            hun = Hungarian()
            pred_bborder, _ = hun(associate_maxtrix, Y_bbvert)
            pred_bborder = Ops.cast(pred_bborder, torch.int32)
            y_bbvert_pred = Ops.gather_tensor_along_2nd_axis(y_bbvert_pred_raw, pred_bborder)

            y_bbscore_pred = Ops.bbscore_association(y_bbscore_pred_raw, pred_bborder)

            # MCE LOSS
            psemce_loss = Ops.get_loss_psem_ce(bat_psem_onehot, y_psem_logits)

            # loss bbox
            bbvert_loss, bbvert_loss_l2, bbvert_loss_ce, bbvert_loss_iou = \
                Ops.get_loss_bbvert(bat_pc, y_bbvert_pred, bat_bbvert, label='use_all_ce_l2_iou')
            bbscore_loss = Ops.get_loss_bbscore(y_bbscore_pred, bat_bbvert)
            # sum_bbox_vert_loss = writer.add_scalar('bbvert_loss', bbvert_loss)
            # sum_bbox_vert_loss_l2 = writer.add_scalar('bbvert_loss_l2', bbvert_loss_l2)
            # sum_bbox_vert_loss_ce = writer.add_scalar('bbvert_loss_ce', bbvert_loss_ce)
            # sum_bbox_vert_loss_iou = writer.add_scalar('bbvert_loss_iou', bbvert_loss_iou)
            # sum_bbox_score_loss = writer.add_scalar('bbscore_loss', bbscore_loss)

            pred_mask_pred = pmask_net(point_features, global_features, y_bbvert_pred_raw, y_bbscore_pred)

            # loss pred_mask
            # pmask_loss = Ops.get_loss_pmask(bat_pc[:, :, 0:9], pred_mask_pred, bat_pmask)
            pmask_loss = FocalLoss2(alpha=0.25, gamma=2, reduce=True)
            ms_loss = pmask_loss(pred_mask_pred, bat_pmask)

            end_2_end_loss = bbvert_loss + bbscore_loss + ms_loss + psemce_loss
            end_2_end_loss.backward()
            optimizer.step()
