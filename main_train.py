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


LOG_DIR = './'
save_model_dir = os.path.join(LOG_DIR, 'checkpoints')
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)

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
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    from dataset import Data_S3DIS as Data

    writer = SummaryWriter('logs')

    train_areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
    test_areas = ['Area_5']
    #
    dataset_path = './Data_S3DIS/'
    # data = S3DISDataset(split='train', data_root=dataset_path, transform=None)
    data = Data(dataset_path, train_areas, test_areas, train_batch_size=64)

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

            total_loss = bbvert_loss + bbscore_loss + ms_loss + psemce_loss
            total_loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print("{}th iteration , loss is : {}".format(i + 1, total_loss))
                torch.save(backbone.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'backbone', i))
                torch.save(bbox_net.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'bbox_net', i))
                torch.save(pmask_net.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'pmask_net', i))

        if epoch % 5 == 0:
            torch.save(backbone.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'backbone', epoch))
            torch.save(bbox_net.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'bbox_net', epoch))
            torch.save(pmask_net.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'pmask_net', epoch))
