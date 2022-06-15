import torch.utils.data
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from BoNetMLP import *
from RandLANet import RandLA, Psem_Loss_ignored
from helper_net import Ops
from main_eval_randla import Evaluation
from dataset_randla_hz import Data_Configs_RandLA

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


today = datetime.now()

if today.hour < 12:
    h = "00"
else:
    h = "12"

is_colab = False
colab_path = '/content/drive/MyDrive/3dbonet_checkpoints/checkpoints/'

LOG_DIR = '' if not is_colab else colab_path
save_model_dir = os.path.join(LOG_DIR, 'checkpoints')
save_model_dir = os.path.join(save_model_dir, today.strftime('%Y%m%d') + h)
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.set_num_threads(4)
    from dataset_randla_hz import Data_S3DIS as Data

    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if not is_colab else colab_path

    summary_path = 'logs_randla' if not is_colab else colab_path +'logs_randla'
    writer = SummaryWriter(summary_path)

    train_areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
    test_areas = ['Area_5']
    #
    dataset_path = './Data_S3DIS_bak/'
    # data = S3DISDataset(split='train', data_root=dataset_path, transform=None)
    batch_size = 8
    data = Data(dataset_path, train_areas, test_areas, train_batch_size=batch_size)

    # train(net, data)

    # some parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # train_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    # train_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints/20220610')

    # backbone_pointnet2
    # backbone = backbone_pointnet2(is_train=True)
    config = Data_Configs_RandLA()
    backbone = RandLA(num_layers=config.num_layers, d_out=config.d_out,num_classes=config.sem_num)
    backbone = backbone.to(device)
    # backbone.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'backbone_out_001.pth')))
    count1 = count_parameters(backbone)

    # sem
    # sem = backbone_sem()
    # sem = sem.to(device)
    # count4 = count_parameters(sem)

    # bbox_net
    bbox_net = bbox_net()
    bbox_net = bbox_net.to(device)
    # bbox_net.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'bbox_net_out_001.pth')))
    count2 = count_parameters(bbox_net)

    # pmask_net
    pmask_net = pmask_net()
    pmask_net = pmask_net.to(device)
    # pmask_net.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'pmask_net_out_001.pth')))
    count3 = count_parameters(pmask_net)

    print('parameters total count : {}'.format(count1 + count2 + count3))

    optim_params = [
        {'params': backbone.parameters(), 'lr': 0.0005, 'betas': (0.9, 0.999), 'eps': 1e-08},
        {'params': bbox_net.parameters(), 'lr': 0.0005, 'betas': (0.9, 0.999), 'eps': 1e-08},
        {'params': pmask_net.parameters(), 'lr': 0.0005, 'betas': (0.9, 0.999), 'eps': 1e-08},
    ]
    optimizer = optim.Adam(optim_params)
    total_train_batch_num = data.total_train_batch_num
    train_old = False
    epoch = -1
    if train_old:
        check_point = torch.load(os.path.join(MODEL_PATH, 'latest_model_30.pt'))
        backbone.load_state_dict(check_point['backbone_state_dict'])
        backbone.train()
        bbox_net.load_state_dict(check_point['bbox_state_dict'])
        bbox_net.train()
        pmask_net.load_state_dict(check_point['pmask_state_dict'])
        pmask_net.train()
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        epoch = check_point['epoch']
        total_loss = check_point['loss']
        print('load net, epoch : {},  total_loss : {}'.format(epoch, total_loss))

    print('total train batch num:', total_train_batch_num)
    for ep in range(epoch + 1, epoch + 51, 1):
        for g in optimizer.param_groups:
            lr = max(0.0005 / (2 ** (ep // 20)), 0.00001)
            g['lr'] = lr
            print('ep : {}, lr : {}'.format(ep, lr))
        data.shuffle_train_files(ep)
        total_loss = 0
        last_ep_time = datetime.now().strftime("%H:%M:%S")
        for i in range(total_train_batch_num):
            batchdata = data.load_train_next_batch_randla()
            bat_bbvert = batchdata['bbvert_padded_labels']
            bat_psem_onehot = batchdata['psem_onehot_labels']
            bat_pmask = batchdata['pmask_padded_labels']
            bat_pc = batchdata['features'].to(device).permute(0, 2, 1).contiguous()
            if torch.cuda.is_available():
                bat_bbvert = bat_bbvert.to(device)
                bat_psem_onehot = bat_psem_onehot.to(device)
                bat_pmask = bat_pmask.to(device)
            # point_features, global_features, y_sem_pred, y_psem_logits = backbone(bat_pc[:, :, 0:9])
            point_features, global_features, _, y_psem_logits = backbone(batchdata, device)

            get_loss_psem_ce = Psem_Loss_ignored(device).cuda()
            # get_loss_psem_ce = PsemCeLoss().cuda()
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
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            if i % 20 == 0:
                print("-----------------------------------------------------------")
                print('bbvert_loss : {}, bbvert_loss_l2 : {}, bbvert_loss_ce : {}, bbvert_loss_iou: {}'.format(
                    bbvert_loss, bbvert_loss_l2, bbvert_loss_ce, bbvert_loss_iou
                ))
                print("-----------------------------------------------------------")
                print("time : {}, {} epoch  {}th iteration , loss is : {}".format(current_time,
                                                                                  ep, i, total_loss))
                print('bbvert_loss: {}, bbscore_loss : {}, ms_loss : {}, psemce_loss : {}'.format(
                    bbvert_loss, bbscore_loss, ms_loss, psemce_loss
                ))
                print("-----------------------------------------------------------")
                x_axis = ep * total_train_batch_num * batch_size + (batch_size * i)
                sum_bbox_vert_loss = writer.add_scalar('bbvert_loss', bbvert_loss, x_axis)
                sum_bbox_vert_loss_l2 = writer.add_scalar('bbvert_loss_l2', bbvert_loss_l2, x_axis)
                sum_bbox_vert_loss_ce = writer.add_scalar('bbvert_loss_ce', bbvert_loss_ce, x_axis)
                sum_bbox_vert_loss_iou = writer.add_scalar('bbvert_loss_iou', bbvert_loss_iou, x_axis)
                sum_bbox_score_loss = writer.add_scalar('bbscore_loss', bbscore_loss, x_axis)
                sum_total_loss = writer.add_scalar('bbscore_loss', total_loss, x_axis)
                sum_pmask_loss = writer.add_scalar('bbscore_loss', ms_loss, x_axis)
                sum_psemce_loss = writer.add_scalar('bbscore_loss', psemce_loss, x_axis)
                # torch.save(backbone.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'backbone', i))
                # torch.save(bbox_net.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'bbox_net', i))
                # torch.save(pmask_net.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'pmask_net', i))
        if ep % 5 == 0:
            print('saving model : ', datetime.now().strftime("%H:%M:%S"))
            PATH = os.path.join(BASE_DIR, save_model_dir, 'latest_model_%s.pt' % ep)
            params = {
                'epoch': ep,
                'backbone_state_dict': backbone.state_dict(),
                'bbox_state_dict': bbox_net.state_dict(),
                'pmask_state_dict': pmask_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }
            torch.save(params, PATH)
            print("saving model successfully : ", datetime.now().strftime("%H:%M:%S"))
            if ep == 0 or ep == 5 or ep == 15 or ep == 25:
                continue
            result_path = './train_evaluate/' + today.strftime('%Y%m%d') + '/' + test_areas[0] + '/'
            print(result_path)
            last_valid_mPre = 0
            with torch.no_grad():
                print('evaluating model : ', datetime.now().strftime("%H:%M:%S"))
                Evaluation.ttest(data, result_path, test_batch_size=batch_size, MODEL_PATH=PATH)
                valid_mPre, valid_mRec = Evaluation.evaluation(dataset_path, train_areas, result_path, writer, ep)
                if valid_mPre >= last_valid_mPre:
                    last_valid_mPre = valid_mPre
                    torch.save(params, '{}/{}_epoch-{}_area-{}_mPre-{}_mRec-{}'.format(os.path.join(BASE_DIR, save_model_dir), 'bonet_pointnet', ep, test_areas[0], valid_mPre, valid_mRec))
                print('evaluation ends : ', datetime.now().strftime("%H:%M:%S"))
        ep_end_time = datetime.now().strftime("%H:%M:%S")
        print('ep {} start time : {}, end time : {}'.format(ep, last_ep_time, ep_end_time))
        last_ep_time = ep_end_time
        print("-----------------------------------------------------------")
        print("--------------------ep ends here----==------------")
        print("-----------------------------------------------------------")
