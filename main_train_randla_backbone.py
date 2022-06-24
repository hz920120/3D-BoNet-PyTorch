import os

import torch.utils.data
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from RandLANet import RandLA, Psem_Loss_ignored
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
colab_path = '/content/drive/MyDrive/3dbonet_checkpoints/'

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
    # from utils.s3dis_loader import S3DISDataset as Data

    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if not is_colab else colab_path

    summary_path = 'logs_randla' if not is_colab else os.path.join(colab_path, 'logs_randla')
    writer = SummaryWriter(summary_path)

    train_areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
    test_areas = ['Area_5']
    #
    dataset_path = './Data_S3DIS/'
    # data = S3DISDataset(split='train', data_root=dataset_path, transform=None)
    batch_size = 4
    data = Data(dataset_path, train_areas, test_areas, train_batch_size=batch_size)

    # train(net, data)

    # some parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODEL_PATH = os.path.join(BASE_DIR, save_model_dir)

    # backbone_pointnet2
    # backbone = backbone_pointnet2(is_train=True)
    config = Data_Configs_RandLA()
    backbone = RandLA(num_layers=config.num_layers, d_out=config.d_out,num_classes=config.sem_num)
    backbone = backbone.to(device)
    # backbone.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'backbone_out_001.pth')))
    count1 = count_parameters(backbone)

    print('parameters total count : {}'.format(count1 ))

    lr_backbone = 0.01
    lr_bbox = 0.0005
    lr_pmask = 0.005
    optim_params = [
        {'params': backbone.parameters(), 'lr': lr_backbone, 'betas': (0.9, 0.999), 'eps': 1e-08, 'name': 'backbone'},
    ]
    optimizer = optim.Adam(optim_params)
    total_train_batch_num = data.total_train_batch_num
    train_old = False
    epoch = -1
    if train_old:
        check_point = torch.load(os.path.join(MODEL_PATH, 'latest_model_45.pt'))
        backbone.load_state_dict(check_point['backbone_state_dict'])
        backbone.train()
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        epoch = check_point['epoch']
        total_loss = check_point['loss']
        print('load net, epoch : {},  total_loss : {}'.format(epoch, total_loss))

    print('total train batch num:', total_train_batch_num)
    for ep in range(epoch + 1, epoch + 100, 1):
        for g in optimizer.param_groups:
            if g['name'] == 'backbone':
                lr = lr_backbone * (0.95 ** ep)
                # lr = max(lr_backbone / (2 ** (ep // 20)), 0.00001)
            elif g['name'] == 'bbox_net':
                lr = max(lr_bbox / (2 ** (ep // 20)), 0.00001)
            else:
                lr = max(lr_pmask / (2 ** (ep // 20)), 0.00001)
            g['lr'] = lr
            # print('ep : {}, lr : {}'.format(ep, lr))
            print('ep : {},      name : {},     lr : {}'.format(ep, g['name'], lr))
        data.shuffle_train_files(ep)
        total_loss = 0
        last_ep_time = datetime.now().strftime("%H:%M:%S")
        for i in range(total_train_batch_num):
            batchdata = data.load_train_next_batch_randla()
            bat_psem_onehot = batchdata['psem_onehot_labels']
            bat_pc = batchdata['features'].to(device).permute(0, 2, 1).contiguous()
            if torch.cuda.is_available():
                bat_psem_onehot = bat_psem_onehot.to(device)
            # point_features, global_features, y_sem_pred, y_psem_logits = backbone(bat_pc[:, :, 0:9])
            point_features, global_features, _, y_psem_logits = backbone(batchdata, device)

            get_loss_psem_ce = Psem_Loss_ignored(device).cuda()
            # get_loss_psem_ce = PsemCeLoss().cuda()
            psemce_loss = get_loss_psem_ce(y_psem_logits, bat_psem_onehot)

            total_loss = psemce_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            if i % 20 == 0:
                print("-----------------------------------------------------------")
                print("time : {}, {} epoch  {}th iteration , loss is : {}".format(current_time,
                                                                                  ep, i, total_loss))
                print('psemce_loss : {}'.format(
                    psemce_loss
                ))
                print("-----------------------------------------------------------")
                x_axis = ep * total_train_batch_num * batch_size + (batch_size * i)
                sum_psemce_loss = writer.add_scalar('bbscore_loss', psemce_loss, x_axis)
                # torch.save(backbone.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'backbone', i))
                # torch.save(bbox_net.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'bbox_net', i))
                # torch.save(pmask_net.state_dict(), '%s/%s_%.3d.pth' % (save_model_dir, 'pmask_net', i))
        if ep % 1 == 0:
            print('saving model : ', datetime.now().strftime("%H:%M:%S"))
            PATH = os.path.join(BASE_DIR, save_model_dir, 'latest_model_%s.pt' % ep)
            params = {
                'epoch': ep,
                'backbone_state_dict': backbone.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }
            torch.save(params, PATH)
            print("saving model successfully : ", datetime.now().strftime("%H:%M:%S"))
            if ep != 100:
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
