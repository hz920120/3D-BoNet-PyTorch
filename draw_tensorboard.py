import os

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from BoNetMLP import *
from dataset import Data_S3DIS as Data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def draw_tensorboard(bbvert_loss, bbvert_loss_l2, bbvert_loss_ce, bbvert_loss_iou,
                     bbscore_loss, ms_loss, psemce_loss, total_loss, writer,
                     epoch, total_train_batch_num, batch_size, i=0):
    print("-----------------------------------------------------------")
    print('bbvert_loss : {}, bbvert_loss_l2 : {}, bbvert_loss_ce : {}, bbvert_loss_iou: {}'.format(
        bbvert_loss, bbvert_loss_l2, bbvert_loss_ce, bbvert_loss_iou
    ))
    print("-----------------------------------------------------------")
    print('bbvert_loss: {}, bbscore_loss : {}, ms_loss : {}, psemce_loss : {}'.format(
        bbvert_loss, bbscore_loss, ms_loss, psemce_loss
    ))
    print("-----------------------------------------------------------")
    x_axis = int(epoch) * total_train_batch_num + (batch_size * i)
    print('x_axis : {}'.format(x_axis))
    writer.add_scalar('bbvert_loss', bbvert_loss, x_axis)
    writer.add_scalar('bbvert_loss_l2', bbvert_loss_l2, x_axis)
    writer.add_scalar('bbvert_loss_ce', bbvert_loss_ce, x_axis)
    writer.add_scalar('bbvert_loss_iou', bbvert_loss_iou, x_axis)
    writer.add_scalar('bbscore_loss', bbscore_loss, x_axis)
    writer.add_scalar('total_loss', total_loss, x_axis)
    writer.add_scalar('ms_loss', ms_loss, x_axis)
    writer.add_scalar('psemce_loss', psemce_loss, x_axis)


def getloss(data, writer, ep, total_train_batch_num, batch_size):
    bat_pc, _, _, bat_psem_onehot, bat_bbvert, bat_pmask = data.load_train_next_batch()
    if torch.cuda.is_available():
        bat_pc, _, _, bat_bbvert, bat_pmask, bat_psem_onehot = \
            torch.from_numpy(bat_pc).cuda(), _, _, torch.from_numpy(bat_bbvert).cuda(), \
            torch.from_numpy(bat_pmask).cuda(), torch.from_numpy(bat_psem_onehot).cuda()
    # point_features, global_features, y_sem_pred, y_psem_logits = backbone(bat_pc[:, :, 0:9])
    point_features, global_features, _, y_psem_logits = backbone(bat_pc[:, :, 0:9])

    get_loss_psem_ce = PsemCeLoss().cuda()
    psemce_loss = get_loss_psem_ce(y_psem_logits, bat_psem_onehot)

    y_bbvert_pred_raw, y_bbscore_pred_raw = bbox_net(global_features)

    associate_maxtrix, Y_bbvert = Ops.bbvert_association(bat_pc, y_bbvert_pred_raw, bat_bbvert,
                                                         label='use_all_ce_l2_iou')
    hun = Hungarian().cuda()
    pred_bborder, _ = hun(associate_maxtrix, Y_bbvert)
    pred_bborder = Ops.cast(pred_bborder, torch.int32)
    y_bbvert_pred = Ops.gather_tensor_along_2nd_axis(y_bbvert_pred_raw, pred_bborder)

    y_bbscore_pred = Ops.bbscore_association(y_bbscore_pred_raw, pred_bborder)

    # loss bbox
    # bbvert_loss, bbvert_loss_l2, bbvert_loss_ce, bbvert_loss_iou = \
    #     Ops.get_loss_bbvert(bat_pc, y_bbvert_pred, bat_bbvert, label='use_all_ce_l2_iou')
    get_loss_bbvert = BbVertLoss('use_all_ce_l2_iou').cuda()
    bbvert_loss, bbvert_loss_l2, bbvert_loss_ce, bbvert_loss_iou = get_loss_bbvert(bat_pc, y_bbvert_pred,
                                                                                   bat_bbvert)
    # bbscore_loss = Ops.get_loss_bbscore(y_bbscore_pred, bat_bbvert)
    get_loss_bbscore = BbScoreLoss().cuda()
    bbscore_loss = get_loss_bbscore(y_bbscore_pred, bat_bbvert)

    # sum_bbox_vert_loss = writer.add_scalar('bbvert_loss', bbvert_loss)
    # sum_bbox_vert_loss_l2 = writer.add_scalar('bbvert_loss_l2', bbvert_loss_l2)
    # sum_bbox_vert_loss_ce = writer.add_scalar('bbvert_loss_ce', bbvert_loss_ce)
    # sum_bbox_vert_loss_iou = writer.add_scalar('bbvert_loss_iou', bbvert_loss_iou)
    # sum_bbox_score_loss = writer.add_scalar('bbscore_loss', bbscore_loss)

    pred_mask_pred = pmask_net(point_features, global_features, y_bbvert_pred, y_bbscore_pred)

    # y_pmask_pred_raw = pmask_net(point_features, global_features, y_bbvert_pred_raw, y_bbscore_pred_raw)

    # loss pred_mask
    # pmask_loss = Ops.get_loss_pmask(bat_pc[:, :, 0:9], pred_mask_pred, bat_pmask)
    pmask_loss = FocalLoss2(alpha=0.75, gamma=2, reduce=True).cuda()
    ms_loss = pmask_loss(bat_pc[:, :, 0:9], pred_mask_pred, bat_pmask)

    total_loss = bbvert_loss + bbscore_loss + ms_loss + psemce_loss
    print('new total loss is : {}'.format(total_loss))
    draw_tensorboard(bbvert_loss, bbvert_loss_l2, bbvert_loss_ce, bbvert_loss_iou,
                     bbscore_loss, ms_loss, psemce_loss, total_loss, writer=writer, epoch=ep,
                     total_train_batch_num=total_train_batch_num, batch_size=batch_size)




if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  ## specify the GPU to use

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
    test_areas = ['Area_5']
    #
    dataset_path = './Data_S3DIS/'
    # data = S3DISDataset(split='train', data_root=dataset_path, transform=None)
    batch_size = 8
    data = Data(dataset_path, train_areas, test_areas, train_batch_size=batch_size)
    # init networks
    backbone = backbone_pointnet2(is_train=True)
    backbone = backbone.to(device)
    bbox_net = bbox_net()
    bbox_net = bbox_net.to(device)
    pmask_net = pmask_net()
    pmask_net = pmask_net.to(device)

    optim_params = [
        {'params': backbone.parameters(), 'lr': 0.0005, 'betas': (0.9, 0.999), 'eps': 1e-08, 'name': 'backbone'},
        {'params': bbox_net.parameters(), 'lr': 0.0005, 'betas': (0.9, 0.999), 'eps': 1e-08, 'name': 'bbox_net'},
        {'params': pmask_net.parameters(), 'lr': 0.0005, 'betas': (0.9, 0.999), 'eps': 1e-08, 'name': 'pmask_net'},
    ]
    optimizer = optim.Adam(optim_params)
    total_train_batch_num = data.total_train_batch_num

    writer = SummaryWriter('logs_temp')

    save_model_dir = os.path.join(BASE_DIR, 'checkpoints/test')
    path_list = os.listdir(save_model_dir)
    path_list.sort(key=lambda x: int(x.split('latest_model_')[1].split('.pt')[0]))
    for i in path_list:
        PATH = os.path.join(save_model_dir, i)
        print(i)
        ep = i.split('latest_model_')[1].split('.pt')[0]
        print(ep)
        MODEL_PATH = os.path.join(BASE_DIR, save_model_dir)
        check_point = torch.load(os.path.join(MODEL_PATH, i))
        backbone.load_state_dict(check_point['backbone_state_dict'])
        backbone.train()
        bbox_net.load_state_dict(check_point['bbox_state_dict'])
        bbox_net.train()
        pmask_net.load_state_dict(check_point['pmask_state_dict'])
        pmask_net.train()
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        epoch = check_point['epoch']
        total_loss = check_point['loss']
        print('load net, epoch : {},  load total_loss : {}'.format(epoch, total_loss))
        getloss(data, writer, ep, total_train_batch_num, batch_size)