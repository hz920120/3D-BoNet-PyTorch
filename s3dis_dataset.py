#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import time
import glob
import pickle
import numpy as np
import copy
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
sys.path.append(base_dir)
sys.path.append(root_dir)

from utils.helper_ply import read_ply
from data_process import DataProcessing

class Data_Configs_RandLA:
    sem_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
                 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
    sem_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    points_cc = 9
    sem_num = len(sem_names)
    ins_max_num = 40
    noise_init = 3.5
    n_pts = 40960
    num_layers = 5
    k_n = 16
    sub_grid_size = 0.04
    sub_sampling_ratio = [4, 4, 4, 4, 2]
    d_out = [16, 64, 128, 256, 512]

class S3DIS(torch_data.Dataset):
    def __init__(self, data_path, mode, test_area_idx=5):
        self.name = 'S3DIS'
        self.mode = mode
        self.path = data_path
        self.label_to_names = {
            0: 'ceiling',
            1: 'floor',
            2: 'wall',
            3: 'beam',
            4: 'column',
            5: 'window',
            6: 'door',
            7: 'table',
            8: 'chair',
            9: 'sofa',
            10: 'bookcase',
            11: 'board',
            12: 'clutter'
        }
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort(
            [k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = glob.glob(
            os.path.join(self.path, 'original_ply', '*.ply'))

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.possibility = {'training': [], 'validation': []}
        self.min_possibility = {'training': [], 'validation': []}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.sub_ins_labels = {'training': [], 'validation': []}
        self.sub_norm_xyz = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}

        # ConfigS3DIS.ignored_label_inds = [
        #     self.label_to_idx[ign_label] for ign_label in self.ignored_labels
        # ]
        self.class_weights = DataProcessing.get_class_weights('S3DIS')
        self.load_sub_sampled_clouds(Data_Configs_RandLA.sub_grid_size, self.mode)
        # self.load_ori_clouds(Data_Configs_RandLA.sub_grid_size, self.mode)
        self.sem_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    def load_sub_sampled_clouds(self, sub_grid_size, mode):
        tree_path = os.path.join(self.path,
                                 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = os.path.join(tree_path,
                                        '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = os.path.join(tree_path,
                                        '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack(
                (data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']
            sub_ins_labels = data['ins_labels']
            sub_xyz = np.vstack(
                (data['x'], data['y'], data['z'])).T

            sub_limit = np.vstack(
                (data['limit_x'], data['limit_y'], data['limit_z'])).T

            norm_xyz = sub_xyz / sub_limit

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.sub_ins_labels[cloud_split] += [sub_ins_labels]
            self.input_names[cloud_split] += [cloud_name]
            self.sub_norm_xyz[cloud_split] += [norm_xyz]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(
                kd_tree_file.split('/')[-1], size * 1e-6,
                time.time() - t0))
        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = os.path.join(tree_path,
                                         '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name,
                                                    time.time() - t0))

        for i, tree in enumerate(self.input_colors[mode]):
            self.possibility[mode].append(
                np.random.rand(tree.data.shape[0]) * 1e-3)  # (0,0.001)
            self.min_possibility[mode].append(
                float(np.min(self.possibility[mode][-1])))

    def __len__(self):
        if self.mode == 'training':
            return len(self.input_trees['training'])
        elif self.mode == 'validation':
            return len(self.input_trees['validation'])

    def __getitem__(self, item):
        pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = self.spatially_regular_gen()

        return pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels

    def spatially_regular_gen(self):
        # Generator loop
        cloud_idx = int(np.argmin(self.min_possibility[self.mode]))
        # choose the point with the minimum of possibility in the cloud as query point
        point_ind = np.argmin(self.possibility[self.mode][cloud_idx])
        # Get all points within the cloud from tree structure
        points = np.array(self.input_trees[self.mode][cloud_idx].data,
                          copy=False)
        # Center point of input region
        center_point = points[point_ind, :].reshape(1, -1)
        # Add noise to the center point
        noise = np.random.normal(scale=Data_Configs_RandLA.noise_init / 10,
                                 size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)
        # Check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < Data_Configs_RandLA.n_pts:
            # Query all points within the cloud
            queried_idx = self.input_trees[self.mode][cloud_idx].query(
                pick_point, k=len(points))[1][0]
        else:
            # Query the predefined number of points
            queried_idx = self.input_trees[self.mode][cloud_idx].query(
                pick_point, k=Data_Configs_RandLA.n_pts)[1][0]

        # Shuffle index
        queried_idx = DataProcessing.shuffle_idx(queried_idx)
        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]
        queried_pc_xyz = queried_pc_xyz - pick_point
        queried_pc_colors = self.input_colors[
            self.mode][cloud_idx][queried_idx]
        queried_pc_labels = self.input_labels[
            self.mode][cloud_idx][queried_idx]
        queried_ins_labels = self.sub_ins_labels[
            self.mode][cloud_idx][queried_idx]
        norm_xyz = self.sub_norm_xyz[
            self.mode][cloud_idx][queried_idx]

        # Update the possibility of the selected points
        dists = np.sum(np.square(
            (points[queried_idx] - pick_point).astype(np.float32)),
                       axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[self.mode][cloud_idx][queried_idx] += delta
        self.min_possibility[self.mode][cloud_idx] = float(
            np.min(self.possibility[self.mode][cloud_idx]))

        # up_sampled with replacement
        if len(points) < Data_Configs_RandLA.n_pts:
            queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels, queried_ins_labels= \
                DataProcessing.data_aug_new(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_ins_labels, queried_idx, Data_Configs_RandLA.n_pts)

        pc_xyzrgb = np.concatenate([queried_pc_xyz, queried_pc_colors], axis=-1)
        sem_labels = queried_pc_labels
        ins_labels = queried_ins_labels

        min_x = np.min(pc_xyzrgb[:, 0])
        max_x = np.max(pc_xyzrgb[:, 0])
        min_y = np.min(pc_xyzrgb[:, 1])
        max_y = np.max(pc_xyzrgb[:, 1])
        min_z = np.min(pc_xyzrgb[:, 2])
        max_z = np.max(pc_xyzrgb[:, 2])

        ori_xyz = copy.deepcopy(pc_xyzrgb[:, 0:3])  # reserved for final visualization
        use_zero_one_center = True
        if use_zero_one_center:
            pc_xyzrgb[:, 0:1] = (pc_xyzrgb[:, 0:1] - min_x) / np.maximum((max_x - min_x), 1e-3)
            pc_xyzrgb[:, 1:2] = (pc_xyzrgb[:, 1:2] - min_y) / np.maximum((max_y - min_y), 1e-3)
            pc_xyzrgb[:, 2:3] = (pc_xyzrgb[:, 2:3] - min_z) / np.maximum((max_z - min_z), 1e-3)

        pc_xyzrgb = np.concatenate([pc_xyzrgb, norm_xyz], axis=-1)
        pc_xyzrgb = np.concatenate([pc_xyzrgb, ori_xyz], axis=-1)

        ########
        sem_labels = sem_labels.reshape([-1])
        ins_labels = ins_labels.reshape([-1])
        bbvert_padded_labels, pmask_padded_labels = self.get_bbvert_pmask_labels(pc_xyzrgb, ins_labels)

        # print('max num is : ' + str(np.max(ins_labels)))

        psem_onehot_labels = np.zeros((pc_xyzrgb.shape[0], Data_Configs_RandLA.sem_num), dtype=np.int8)
        for idx, s in enumerate(sem_labels):
            if sem_labels[idx] == -1: continue  # invalid points
            sem_idx = Data_Configs_RandLA.sem_ids.index(s)
            psem_onehot_labels[idx, sem_idx] = 1

        return pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels


    @staticmethod
    def get_bbvert_pmask_labels(pc, ins_labels):
        gt_bbvert_padded = np.zeros((Data_Configs_RandLA.ins_max_num, 2, 3), dtype=np.float32)
        gt_pmask = np.zeros((Data_Configs_RandLA.ins_max_num, pc.shape[0]), dtype=np.float32)
        count = -1
        unique_ins_labels = np.unique(ins_labels)
        for ins_ind in unique_ins_labels:
            if ins_ind <= -1: continue
            count += 1
            if count >= Data_Configs_RandLA.ins_max_num: print('ignored! more than max instances:',
                                                               len(unique_ins_labels)); continue

            ins_labels_tp = np.zeros(ins_labels.shape, dtype=np.int8)
            ins_labels_tp[ins_labels == ins_ind] = 1
            ins_labels_tp = np.reshape(ins_labels_tp, [-1])
            gt_pmask[count, :] = ins_labels_tp

            ins_labels_tp_ind = np.argwhere(ins_labels_tp == 1)
            ins_labels_tp_ind = np.reshape(ins_labels_tp_ind, [-1])

            ###### bb min_xyz, max_xyz
            pc_xyz_tp = pc[:, 0:3]
            pc_xyz_tp = pc_xyz_tp[ins_labels_tp_ind]
            gt_bbvert_padded[count, 0, 0] = x_min = np.min(pc_xyz_tp[:, 0])
            gt_bbvert_padded[count, 0, 1] = y_min = np.min(pc_xyz_tp[:, 1])
            gt_bbvert_padded[count, 0, 2] = z_min = np.min(pc_xyz_tp[:, 2])
            gt_bbvert_padded[count, 1, 0] = x_max = np.max(pc_xyz_tp[:, 0])
            gt_bbvert_padded[count, 1, 1] = y_max = np.max(pc_xyz_tp[:, 1])
            gt_bbvert_padded[count, 1, 2] = z_max = np.max(pc_xyz_tp[:, 2])

        return gt_bbvert_padded, gt_pmask


    def tf_map(self, pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels):
        features = pc_xyzrgb
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        batch_pc = pc_xyzrgb[:, :, :3]
        for i in range(Data_Configs_RandLA.num_layers):
            neighbors_idx = DataProcessing.knn_search(batch_pc, batch_pc, Data_Configs_RandLA.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] // Data_Configs_RandLA.sub_sampling_ratio[i], :]
            pool_idx = neighbors_idx[:, :batch_pc.shape[1] // Data_Configs_RandLA.sub_sampling_ratio[i], :]
            up_idx = DataProcessing.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbors_idx)
            input_pools.append(pool_idx)
            input_up_samples.append(up_idx)
            batch_pc = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [features, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels]

        return input_list

    def collate_fn(self, batch):
        pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = [], [], [], [], [], []
        for i in range(len(batch)):
            pc_xyzrgb.append(batch[i][0])
            sem_labels.append(batch[i][1])
            ins_labels.append(batch[i][2])
            psem_onehot_labels.append(batch[i][3])
            bbvert_padded_labels.append(batch[i][4])
            pmask_padded_labels.append(batch[i][5])

        pc_xyzrgb = np.stack(pc_xyzrgb)
        sem_labels = np.stack(sem_labels)
        ins_labels = np.stack(ins_labels)
        psem_onehot_labels = np.stack(psem_onehot_labels)
        bbvert_padded_labels = np.stack(bbvert_padded_labels)
        pmask_padded_labels = np.stack(pmask_padded_labels)

        flat_inputs = self.tf_map(pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels,
                                  pmask_padded_labels)

        num_layers = Data_Configs_RandLA.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append((torch.from_numpy(tmp).float()))
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()  # B,C,N
        inputs['sem_labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['ins_labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['psem_onehot_labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).int()
        inputs['bbvert_padded_labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 4]).float()
        inputs['pmask_padded_labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 5]).float()
        return inputs