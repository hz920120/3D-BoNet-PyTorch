import numpy as np
import scipy.stats
import os
import scipy.io
import torch
import glob
import h5py, sys
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from helper_net import Ops

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def Get_instance_idx(bbox_center, max_output_size, batch_pc, batch_sem, sem_radius):
    batch_size = bbox_center.shape[0]
    predict = []
    batch_sem = torch.Tensor(batch_sem).cuda()
    batch_sem = batch_sem.int()

    for i in range(batch_size):
        bat_pc = batch_pc[i]
        box_center = bbox_center[i]

        for j in range(max_output_size):
            idx = torch.argmax(box_center).long()
            sem = batch_sem[idx]
            if (box_center[idx] <= 0.3):
                break

            else:

                dist = (torch.sum((bat_pc[idx, 0:3] - bat_pc[:, 0:3]) ** 2, dim=1)) ** 0.5
                need_zero = (dist <= sem_radius[sem]) & (batch_sem == sem)
                box_center[need_zero] = 0

                predict.append(idx)

    return torch.tensor(predict)


class Eval_Tools:
    @staticmethod
    def get_scene_list(res_blocks):
        scene_list_dic = {}
        for b in res_blocks:
            scene_name = b.split('/')[-1][0:-len('_0000')]
            if scene_name not in scene_list_dic: scene_list_dic[scene_name] = []
            scene_list_dic[scene_name].append(b)
        if len(scene_list_dic) == 0:
            print('scene len is 0, error!');
            exit()
        return scene_list_dic

    @staticmethod
    def get_sem_for_ins(ins_by_pts, sem_by_pts):
        ins_cls_dic = {}
        ins_idx, cnt = np.unique(ins_by_pts, return_counts=True)
        for ins_id, cn in zip(ins_idx, cnt):
            if ins_id == -1: continue  # empty ins
            temp = sem_by_pts[np.argwhere(ins_by_pts == ins_id)][:, 0]
            sem_for_this_ins = scipy.stats.mode(temp)[0][0]
            ins_cls_dic[ins_id] = sem_for_this_ins
        return ins_cls_dic

    @staticmethod
    def BlockMerging(volume, volume_seg, pts, grouplabel, groupseg, gap=1e-3):
        overlapgroupcounts = np.zeros([100, 1000])
        groupcounts = np.ones(100)
        x = (pts[:, 0] / gap).astype(np.int32)
        y = (pts[:, 1] / gap).astype(np.int32)
        z = (pts[:, 2] / gap).astype(np.int32)
        for i in range(pts.shape[0]):
            xx = x[i]
            yy = y[i]
            zz = z[i]
            if grouplabel[i] != -1:
                if volume[xx, yy, zz] != -1 and volume_seg[xx, yy, zz] == groupseg[grouplabel[i]]:
                    overlapgroupcounts[grouplabel[i], volume[xx, yy, zz]] += 1
            groupcounts[grouplabel[i]] += 1

        groupcate = np.argmax(overlapgroupcounts, axis=1)
        maxoverlapgroupcounts = np.max(overlapgroupcounts, axis=1)
        curr_max = np.max(volume)
        for i in range(groupcate.shape[0]):
            if maxoverlapgroupcounts[i] < 7 and groupcounts[i] > 12:
                curr_max += 1
                groupcate[i] = curr_max

        finalgrouplabel = -1 * np.ones(pts.shape[0])
        for i in range(pts.shape[0]):
            if grouplabel[i] != -1 and volume[x[i], y[i], z[i]] == -1:
                volume[x[i], y[i], z[i]] = groupcate[grouplabel[i]]
                volume_seg[x[i], y[i], z[i]] = groupseg[grouplabel[i]]
                finalgrouplabel[i] = groupcate[grouplabel[i]]
        return finalgrouplabel

    @staticmethod
    def get_mean_insSize_by_sem(dataset_path, train_areas):
        from helper_data_s3dis import Data_Configs as Data_Configs
        configs = Data_Configs()

        mean_insSize_by_sem = {}
        for sem in configs.sem_ids: mean_insSize_by_sem[sem] = []

        for a in train_areas:
            print('get mean insSize, check train area:', a)
            files = sorted(glob.glob(dataset_path + a + '*.h5'))
            for file_path in files:
                fin = h5py.File(file_path, 'r')
                semIns_labels = fin['labels'][:].reshape([-1, 2])
                ins_labels = semIns_labels[:, 1]
                sem_labels = semIns_labels[:, 0]

                ins_idx = np.unique(ins_labels)
                for ins_id in ins_idx:
                    tmp = (ins_labels == ins_id)
                    sem = scipy.stats.mode(sem_labels[tmp])[0][0]
                    mean_insSize_by_sem[sem].append(np.sum(np.asarray(tmp, dtype=np.float32)))

        for sem in mean_insSize_by_sem: mean_insSize_by_sem[sem] = np.mean(mean_insSize_by_sem[sem])

        return mean_insSize_by_sem


class Evaluation:
    @staticmethod
    def load_data(dataset_path, train_areas, test_areas):

        ####### 3. load data
        from helper_data_s3dis import Data_S3DIS as Data
        data = Data(dataset_path, train_areas, test_areas)

        return data

    # use GT semantic now (for validation)
    @staticmethod
    def ttest(data, result_path, test_batch_size=1):
        # parameter
        global _
        num_feature = 128
        max_output_size = 64
        # load trained model
        from BoNetMLP import backbone_pointnet2, pmask_net, bbox_net, Hungarian

        # date = '20200620_085341_Area_5'
        # epoch_num = '075'
        MODEL_PATH = os.path.join(BASE_DIR, 'checkpoints')

        backbone_pointnet2 = backbone_pointnet2(is_train=False).cuda()
        backbone_pointnet2.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'backbone_010.pth')))
        backbone_pointnet2 = backbone_pointnet2.eval()

        pmask_net = pmask_net(num_feature).cuda()
        pmask_net.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'pmask_net_010.pth')))
        pmask_net = pmask_net.eval()

        bbox_net = bbox_net().cuda()
        bbox_net.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'bbox_net_010.pth')))
        bbox_net = bbox_net.eval()

        print("Load model suceessfully.")

        test_files = data.test_files
        print('total_test_batch_num_sq', len(test_files))
        scene_list_dic = Eval_Tools.get_scene_list(test_files)

        for scene_name in scene_list_dic:
            # todo path
            print('test scene:', scene_name)
            scene_result = {}
            scene_files = scene_list_dic[scene_name]
            for k in range(0, len(scene_files), test_batch_size):
                t_files = scene_files[k: k + test_batch_size]
                bat_pc, bat_sem_gt, bat_ins_gt, bat_psem_onehot, bat_bbvert, bat_pmask, bat_files = \
                    data.load_test_next_batch_sq(bat_files=t_files)

                if torch.cuda.is_available():
                    bat_pc, bat_bbvert, bat_pmask, bat_psem_onehot = \
                        torch.tensor(bat_pc).cuda(), torch.tensor(bat_bbvert).cuda(), torch.tensor(bat_pmask).cuda(), torch.tensor(bat_psem_onehot).cuda()

                point_features, global_features, y_sem_pred, y_psem_logits = backbone_pointnet2(bat_pc[:, :, 0:3],
                                                                                      bat_pc[:, :, 3:9].transpose(1, 2))

                y_bbvert_pred_sq_raw, y_bbscore_pred_sq_raw = bbox_net(global_features)

                # predict score
                associate_maxtrix, Y_bbvert = Ops.bbvert_association(bat_pc, y_bbvert_pred_sq_raw, bat_bbvert,
                                                                     label='use_all_ce_l2_iou')
                hun = Hungarian()
                pred_bborder, _ = hun(associate_maxtrix, Y_bbvert)
                pred_bborder = Ops.cast(pred_bborder, torch.int32)
                y_bbscore_pred = Ops.bbscore_association(y_bbscore_pred_sq_raw, pred_bborder)
                # predict mask
                pre_mask = pmask_net(point_features, global_features, y_bbvert_pred_sq_raw, y_bbscore_pred)

                for b in range(len(t_files)):
                    pc = np.asarray(bat_pc.cpu().detach()[b], dtype=np.float16)
                    sem_gt = np.asarray(bat_sem_gt[b], dtype=np.int16)
                    ins_gt = np.asarray(bat_ins_gt[b], dtype=np.int32)
                    sem_pred_raw = np.asarray(y_psem_logits.cpu().detach()[b], dtype=np.float16)
                    bbvert_pred_raw = np.asarray(y_bbvert_pred_sq_raw.cpu().detach()[b], dtype=np.float16)
                    bbscore_pred_raw = np.asarray(y_bbscore_pred_sq_raw.cpu().detach()[b], dtype=np.float16)
                    pmask_pred_raw = np.asarray(pre_mask.cpu().detach()[b], dtype=np.float16)

                    block_name = t_files[b][-len('0000'):]
                    scene_result['block_' + block_name] = {'pc': pc, 'sem_gt': sem_gt, 'ins_gt': ins_gt,
                                                           'sem_pred_raw': sem_pred_raw,
                                                           'bbvert_pred_raw': bbvert_pred_raw,
                                                           'bbscore_pred_raw': bbscore_pred_raw,
                                                           'pmask_pred_raw': pmask_pred_raw}

            if len(scene_result) != len(scene_files): print('file testing error'); exit()
            if not os.path.exists(result_path + 'res_by_scene/'): os.makedirs(result_path + 'res_by_scene/')
            scipy.io.savemat(result_path + 'res_by_scene/' + scene_name + '.mat', scene_result, do_compression=True)

    @staticmethod
    def evaluation(dataset_path, train_areas, result_path):
        from helper_data_s3dis import Data_Configs as Data_Configs
        configs = Data_Configs()
        mean_insSize_by_sem = Eval_Tools.get_mean_insSize_by_sem(dataset_path, train_areas)

        TP_FP_Total = {}
        for sem_id in configs.sem_ids:
            TP_FP_Total[sem_id] = {}
            TP_FP_Total[sem_id]['TP'] = 0
            TP_FP_Total[sem_id]['FP'] = 0
            TP_FP_Total[sem_id]['Total'] = 0

        res_scenes = sorted(os.listdir(result_path + 'res_by_scene/'))
        for scene_name in res_scenes:
            print('eval scene', scene_name)
            scene_result = scipy.io.loadmat(result_path + 'res_by_scene/' + scene_name,
                                            verify_compressed_data_integrity=False)

            pc_all = [];
            ins_gt_all = [];
            sem_pred_all = [];
            sem_gt_all = []
            gap = 5e-3
            volume_num = int(1. / gap) + 2
            volume = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)
            volume_sem = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)

            for i in range(len(scene_result)):
                block = 'block_' + str(i).zfill(4)
                if block not in scene_result: continue
                pc = scene_result[block][0]['pc'][0]
                ins_gt = scene_result[block][0]['ins_gt'][0][0]
                sem_gt = scene_result[block][0]['sem_gt'][0][0]
                bbscore_pred_raw = scene_result[block][0]['bbscore_pred_raw'][0][0]
                pmask_pred_raw = scene_result[block][0]['pmask_pred_raw'][0]
                sem_pred_raw = scene_result[block][0]['sem_pred_raw'][0]

                sem_pred = np.argmax(sem_pred_raw, axis=-1)
                pmask_pred = pmask_pred_raw * np.tile(bbscore_pred_raw[:, None], [1, pmask_pred_raw.shape[-1]])
                ins_pred = np.argmax(pmask_pred, axis=-2)
                ins_sem_dic = Eval_Tools.get_sem_for_ins(ins_by_pts=ins_pred, sem_by_pts=sem_pred)
                Eval_Tools.BlockMerging(volume, volume_sem, pc[:, 6:9], ins_pred, ins_sem_dic, gap)

                pc_all.append(pc)
                ins_gt_all.append(ins_gt)
                sem_pred_all.append(sem_pred)
                sem_gt_all.append(sem_gt)
            ##
            pc_all = np.concatenate(pc_all, axis=0)
            ins_gt_all = np.concatenate(ins_gt_all, axis=0)
            sem_pred_all = np.concatenate(sem_pred_all, axis=0)
            sem_gt_all = np.concatenate(sem_gt_all, axis=0)

            pc_xyz_int = (pc_all[:, 6:9] / gap).astype(np.int32)
            ins_pred_all = volume[tuple(pc_xyz_int.T)]

            #### if you need to visulize, please uncomment the follow lines
            # from helper_data_plot import Plot as Plot
            # Plot.draw_pc(np.concatenate([pc_all[:,9:12], pc_all[:,3:6]], axis=1))
            # Plot.draw_pc_semins(pc_xyz=pc_all[:, 9:12], pc_semins=ins_gt_all)
            # Plot.draw_pc_semins(pc_xyz=pc_all[:, 9:12], pc_semins=ins_pred_all)
            # Plot.draw_pc_semins(pc_xyz=pc_all[:, 9:12], pc_semins=sem_gt_all)
            # Plot.draw_pc_semins(pc_xyz=pc_all[:, 9:12], pc_semins=sem_pred_all)
            ####

            ###################
            # pred ins
            ins_pred_by_sem = {}
            for sem in configs.sem_ids: ins_pred_by_sem[sem] = []
            ins_idx, cnts = np.unique(ins_pred_all, return_counts=True)
            for ins_id, cn in zip(ins_idx, cnts):
                if ins_id <= -1: continue
                tmp = (ins_pred_all == ins_id)
                sem = scipy.stats.mode(sem_pred_all[tmp])[0][0]
                if cn <= 0.3 * mean_insSize_by_sem[sem]: continue  # remove small instances
                ins_pred_by_sem[sem].append(tmp)
            # gt ins
            ins_gt_by_sem = {}
            for sem in configs.sem_ids: ins_gt_by_sem[sem] = []
            ins_idx = np.unique(ins_gt_all)
            for ins_id in ins_idx:
                if ins_id <= -1: continue
                tmp = (ins_gt_all == ins_id)
                sem = scipy.stats.mode(sem_gt_all[tmp])[0][0]
                if len(np.unique(sem_gt_all[ins_gt_all == ins_id])) != 1: print('sem ins label error'); exit()
                ins_gt_by_sem[sem].append(tmp)
            # to associate
            for sem_id, sem_name in zip(configs.sem_ids, configs.sem_names):
                ins_pred_tp = ins_pred_by_sem[sem_id]
                ins_gt_tp = ins_gt_by_sem[sem_id]

                flag_pred = np.zeros(len(ins_pred_tp), dtype=np.int8)
                for i_p, ins_p in enumerate(ins_pred_tp):
                    iou_max = -1
                    for i_g, ins_g in enumerate(ins_gt_tp):
                        u = ins_g | ins_p
                        i = ins_g & ins_p
                        iou_tp = float(np.sum(i)) / (np.sum(u) + 1e-8)
                        if iou_tp > iou_max:
                            iou_max = iou_tp
                    if iou_max >= 0.5:
                        flag_pred[i_p] = 1
                ###
                TP_FP_Total[sem_id]['TP'] += np.sum(flag_pred)
                TP_FP_Total[sem_id]['FP'] += len(flag_pred) - np.sum(flag_pred)
                TP_FP_Total[sem_id]['Total'] += len(ins_gt_tp)

        ###############
        pre_all = []
        rec_all = []
        for sem_id, sem_name in zip(configs.sem_ids, configs.sem_names):
            TP = TP_FP_Total[sem_id]['TP']
            FP = TP_FP_Total[sem_id]['FP']
            Total = TP_FP_Total[sem_id]['Total']
            pre = float(TP) / (TP + FP + 1e-8)
            rec = float(TP) / (Total + 1e-8)
            if Total > 0:
                pre_all.append(pre)
                rec_all.append(rec)
            out_file = result_path + 'PreRec_' + str(sem_id).zfill(2) + '_' + sem_name + '_' + str(
                round(pre, 4)) + '_' + str(round(rec, 4))
            np.savez_compressed(out_file + '.npz', tp={0, 0})
        out_file = result_path + 'PreRec_mean_' + str(round(np.mean(pre_all), 4)) + '_' + str(
            round(np.mean(rec_all), 4))
        np.savez_compressed(out_file + '.npz', tp={0, 0})

        return 0


#######################
if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  ## specify the GPU to use

    dataset_path = './Data_S3DIS_bak/'
    train_areas = ['Area_1', 'Area_6', 'Area_3', 'Area_2', 'Area_4']
    test_areas = ['Area_5']
    result_path = './log2_radius/test_res/' + test_areas[0] + '/'

    os.system('rm -rf %s' % (result_path))

    data = Evaluation.load_data(dataset_path, train_areas, test_areas)
    Evaluation.ttest(data, result_path, test_batch_size=1)
    Evaluation.evaluation(dataset_path, train_areas, result_path)  # train_areas is just for a parameter
