import glob
import os
import pickle

import numpy as np
import random
import copy
from random import shuffle
import h5py
import torch
from data_process import DataProcessing
from utils.helper_ply import read_ply
from itertools import islice


class Data_Configs_RandLA:
    sem_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
                 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
    sem_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    points_cc = 9
    sem_num = len(sem_names)
    ins_max_num = 24
    noise_init = 3.5
    n_pts = 40960
    num_layers = 5
    k_n = 16
    sub_sampling_ratio = [4, 4, 4, 4, 2]
    d_out = [16, 64, 128, 256, 512]


class Data_S3DIS:
    def __init__(self, dataset_path, train_areas, test_areas, sub_grid_size=0.04, is_train=True, train_batch_size=4):
        self.trees = {'train': [], 'valid': []}
        self.colors = {'train': [], 'valid': []}
        self.labels = {'train': [], 'valid': []}
        self.ins_labels = {'train': [], 'valid': []}
        self.split = 'train' if is_train else 'valid'
        self.noise_init = Data_Configs_RandLA.noise_init
        self.n_pts = Data_Configs_RandLA.n_pts
        self.val_proj = []
        self.val_proj_labels = []
        self.val_proj_ins_labels = []
        self.root_folder_4_traintest = dataset_path
        self.sub_grid_size = sub_grid_size
        self.tree_path = os.path.join(dataset_path, 'input_{:.3f}'.format(self.sub_grid_size))
        self.train_files = self.load_full_file_list(areas=train_areas)
        self.test_files = self.load_full_file_list(areas=test_areas, is_train=False)
        print('train files:', len(self.train_files))
        print('test files:', len(self.test_files))


        self.ins_max_num = Data_Configs_RandLA.ins_max_num
        self.train_batch_size = train_batch_size
        # self.total_train_batch_num = len(self.train_files) // self.train_batch_size
        self.total_train_batch_num = len(self.train_files)

        self.num_layers = Data_Configs_RandLA.num_layers
        self.k_n = Data_Configs_RandLA.k_n
        self.sub_sampling_ratio = Data_Configs_RandLA.sub_sampling_ratio

        self.train_next_bat_index = 0
        self.possibility = {'train': [], 'valid': []}
        self.min_possibility = {'train': [], 'valid': []}

        for i, tree in enumerate(self.colors[self.split]):
            # 第i个点云的颜色数据，每个点都随机了一个possibility值
            #print(np.random.rand(tree.data.shape[0]))
            self.possibility[self.split].append(np.random.rand(tree.data.shape[0]) * 1e-3)  # (0,0.001)
            self.min_possibility[self.split].append(float(np.min(self.possibility[self.split][-1])))
            # 对刚加进来的点云的possibility值，找出最小值来，这样的话每个点云，都有一个min_possibility

    def load_full_file_list(self, areas, is_train=True):
        all_files = []
        for a in areas:
            print('check area:', a)
            file_path = os.path.join(self.root_folder_4_traintest, 'original_ply', a + '*.ply')
            files = sorted(glob.glob(file_path))
            for i, f in enumerate(files):
                name = str(f) + '_' + str(i).zfill(4)
                data = read_ply(f)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                sub_labels = data['class']
                ins_labels = data['ins_labels']
                cloud_name = f.split('/')[-1][:-4]
                kd_tree_file = os.path.join(self.tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
                with open(kd_tree_file, 'rb') as f_tree:
                    search_tree = pickle.load(f_tree)

                cloud_split = 'valid'
                if is_train:
                    cloud_split = 'train'
                self.trees[cloud_split].append(search_tree)
                self.colors[cloud_split].append(sub_colors)
                self.labels[cloud_split].append(sub_labels)
                self.ins_labels[cloud_split].append(ins_labels)

                if not is_train:
                    proj_file = os.path.join(self.tree_path, '{:s}_proj.pkl'.format(cloud_name))
                    with open(proj_file, 'rb') as f:
                        proj_idx, proj_labels = pickle.load(f)
                    self.val_proj.append(proj_idx)
                    self.val_proj_labels.append(proj_labels)
                    self.val_proj_ins_labels.append(proj_labels)
                all_files.append(name)

        return all_files


    def load_full_file_list_new(self, areas):
        all_files = []
        for a in areas:
            print('check area:', a)
            files = sorted(glob.glob(self.root_folder_4_traintest + a + '*.h5'))
            for f in files:
                fin = h5py.File(f, 'r')
                coords = fin['coords'][:]
                semIns_labels = fin['labels'][:].reshape([-1, 2])
                ins_labels = semIns_labels[:, 1]
                sem_labels = semIns_labels[:, 0]

                data_valid = True
                ins_idx = np.unique(ins_labels)
                for i_i in ins_idx:
                    if i_i <= -1: continue
                    sem_labels_tp = sem_labels[ins_labels == i_i]
                    unique_sem_labels = np.unique(sem_labels_tp)
                    if len(unique_sem_labels) >= 2:
                        print('>= 2 sem for an ins:', f)
                        data_valid = False
                        break
                if not data_valid: continue
                block_num = coords.shape[0]
                for b in range(block_num):
                    all_files.append(f + '_' + str(b).zfill(4))

        return np.array_split(all_files, len(all_files) // self.batch)


    @staticmethod
    def load_raw_data_file_s3dis_block(file_path):
        block_id = int(file_path[-4:])
        file_path = file_path[0:-5]

        fin = h5py.File(file_path, 'r')
        coords = fin['coords'][block_id]
        points = fin['points'][block_id]
        semIns_labels = fin['labels'][block_id]

        pc = np.concatenate([coords, points[:, 3:9]], axis=-1)
        sem_labels = semIns_labels[:, 0]
        ins_labels = semIns_labels[:, 1]

        ## if u need to visulize data, uncomment the following lines
        # from helper_data_plot import Plot as Plot
        # Plot.draw_pc(pc)
        # Plot.draw_pc_semins(pc_xyz=pc[:, 0:3], pc_semins=sem_labels, fix_color_num=13)
        # Plot.draw_pc_semins(pc_xyz=pc[:, 0:3], pc_semins=ins_labels)

        return pc, sem_labels, ins_labels


    @staticmethod
    def load_raw_data_file_s3dis_block_new(file_path, is_train=True):

        if not is_train:
            block_id = int(file_path[-4:])
            file_path = file_path[0:-5]

            fin = h5py.File(file_path, 'r')
            coords = fin['coords'][block_id]
            points = fin['points'][block_id]
            semIns_labels = fin['labels'][block_id]

            pc = np.concatenate([coords, points[:, 3:9]], axis=-1)
            sem_labels = semIns_labels[:, 0]
            ins_labels = semIns_labels[:, 1]

            ## if u need to visulize data, uncomment the following lines
            # from helper_data_plot import Plot as Plot
            # Plot.draw_pc(pc)
            # Plot.draw_pc_semins(pc_xyz=pc[:, 0:3], pc_semins=sem_labels, fix_color_num=13)
            # Plot.draw_pc_semins(pc_xyz=pc[:, 0:3], pc_semins=ins_labels)

            return pc, sem_labels, ins_labels


        pc, sem_labels, ins_labels = None, None, None

        for i in file_path:
            block_id = int(i[-4:])
            i = i[0:-5]

            fin = h5py.File(i, 'r')
            coords = fin['coords'][block_id]
            points = fin['points'][block_id]
            semIns_labels = fin['labels'][block_id]

            pc = np.concatenate([coords, points[:, 3:9]], axis=-1) if pc is None else np.append(pc, np.concatenate([coords, points[:, 3:9]], axis=-1), axis=0)
            sem_labels = semIns_labels[:, 0] if sem_labels is None else np.append(sem_labels, semIns_labels[:, 0], axis=0)
            ins_labels = semIns_labels[:, 1] if ins_labels is None else np.append(ins_labels, semIns_labels[:, 1], axis=0)

        ## if u need to visulize data, uncomment the following lines
        # from helper_data_plot import Plot as Plot
        # Plot.draw_pc(pc)
        # Plot.draw_pc_semins(pc_xyz=pc[:, 0:3], pc_semins=sem_labels, fix_color_num=13)
        # Plot.draw_pc_semins(pc_xyz=pc[:, 0:3], pc_semins=ins_labels)

        return pc, sem_labels, ins_labels


    @staticmethod
    def load_raw_data_file_s3dis_block_eval(file_path):
        block_id = int(file_path[-4:])
        file_path = file_path[0:-5]

        fin = h5py.File(file_path, 'r')
        coords = fin['coords'][block_id]
        points = fin['points'][block_id]
        semIns_labels = fin['labels'][block_id]

        pc = np.concatenate([coords, points[:, 3:9]], axis=-1)
        sem_labels = semIns_labels[:, 0]
        ins_labels = semIns_labels[:, 1]

        ## if u need to visulize data, uncomment the following lines
        # from helper_data_plot import Plot as Plot
        # Plot.draw_pc(pc)
        # Plot.draw_pc_semins(pc_xyz=pc[:, 0:3], pc_semins=sem_labels, fix_color_num=13)
        # Plot.draw_pc_semins(pc_xyz=pc[:, 0:3], pc_semins=ins_labels)

        return pc, sem_labels, ins_labels

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

    @staticmethod
    def load_fixed_points(file_path, is_train=True):
        # pc_xyzrgb, sem_labels, ins_labels = Data_S3DIS.load_raw_data_file_s3dis_block(file_path, is_train)
        pc_xyzrgb, sem_labels, ins_labels = Data_S3DIS.load_raw_data_file_s3dis_block(file_path)
        ### center xy within the block
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

        pc_xyzrgb = np.concatenate([pc_xyzrgb, ori_xyz], axis=-1)

        ########
        sem_labels = sem_labels.reshape([-1])
        ins_labels = ins_labels.reshape([-1])
        bbvert_padded_labels, pmask_padded_labels = Data_S3DIS.get_bbvert_pmask_labels(pc_xyzrgb, ins_labels)

        psem_onehot_labels = np.zeros((pc_xyzrgb.shape[0], Data_Configs_RandLA.sem_num), dtype=np.int8)
        for idx, s in enumerate(sem_labels):
            if sem_labels[idx] == -1: continue  # invalid points
            sem_idx = Data_Configs_RandLA.sem_ids.index(s)
            psem_onehot_labels[idx, sem_idx] = 1

        return pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels

    @staticmethod
    def load_fixed_points_randla(file_path, is_train=True):
        pc_xyzrgb, sem_labels, ins_labels = Data_S3DIS.load_raw_data_file_s3dis_block(file_path)

        coord_mean = np.mean(pc_xyzrgb[:, :3], axis=0)
        pc_xyzrgb[:, :3] -= coord_mean
        pc_xyzrgb[:, 3:6] *= 255
        pc_xyzrgb[:, 3:6] /= 127.5 - 1


        # ori_xyz = copy.deepcopy(pc_xyzrgb[:, 0:3])  # reserved for final visualization
        # use_zero_one_center = True
        # if use_zero_one_center:
        #     pc_xyzrgb[:, 0:1] = (pc_xyzrgb[:, 0:1] - min_x) / np.maximum((max_x - min_x), 1e-3)
        #     pc_xyzrgb[:, 1:2] = (pc_xyzrgb[:, 1:2] - min_y) / np.maximum((max_y - min_y), 1e-3)
        #     pc_xyzrgb[:, 2:3] = (pc_xyzrgb[:, 2:3] - min_z) / np.maximum((max_z - min_z), 1e-3)

        # pc_xyzrgb = np.concatenate([pc_xyzrgb, ori_xyz], axis=-1)


        ########
        sem_labels = sem_labels.reshape([-1])
        ins_labels = ins_labels.reshape([-1])
        bbvert_padded_labels, pmask_padded_labels = Data_S3DIS.get_bbvert_pmask_labels(pc_xyzrgb, ins_labels)

        psem_onehot_labels = np.zeros((pc_xyzrgb.shape[0], Data_Configs_RandLA.sem_num), dtype=np.int8)
        for idx, s in enumerate(sem_labels):
            if sem_labels[idx] == -1: continue  # invalid points
            sem_idx = Data_Configs_RandLA.sem_ids.index(s)
            psem_onehot_labels[idx, sem_idx] = 1

        return pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels

    def load_train_next_batch(self):
        bat_files = self.train_files[self.train_next_bat_index *
                                     self.train_batch_size:( self.train_next_bat_index + 1) * self.train_batch_size]
        bat_pc = []
        bat_sem_labels = []
        bat_ins_labels = []
        bat_psem_onehot_labels = []
        bat_bbvert_padded_labels = []
        bat_pmask_padded_labels = []
        for file in bat_files:
            pc, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = Data_S3DIS.load_fixed_points(
                file)
            bat_pc.append(pc)
            bat_sem_labels.append(sem_labels)
            bat_ins_labels.append(ins_labels)
            bat_psem_onehot_labels.append(psem_onehot_labels)
            bat_bbvert_padded_labels.append(bbvert_padded_labels)
            bat_pmask_padded_labels.append(pmask_padded_labels)

        bat_pc = np.asarray(bat_pc, dtype=np.float32)
        bat_sem_labels = np.asarray(bat_sem_labels, dtype=np.float32)
        bat_ins_labels = np.asarray(bat_ins_labels, dtype=np.float32)
        bat_psem_onehot_labels = np.asarray(bat_psem_onehot_labels, dtype=np.float32)
        bat_bbvert_padded_labels = np.asarray(bat_bbvert_padded_labels, dtype=np.float32)
        bat_pmask_padded_labels = np.asarray(bat_pmask_padded_labels, dtype=np.float32)

        self.train_next_bat_index += 1
        return bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot_labels, bat_bbvert_padded_labels, bat_pmask_padded_labels

    def load_train_next_batch_randla(self):
        # Choose the cloud with the lowest probability
        # 根据min_possibility的值选一个点云出来
        cloud_idx = int(np.argmin(self.min_possibility[self.split]))

        # Choose the point with the minimum of possibility in the cloud as query point
        # 对于找出来的点云的所有点，根据possibility的值找出一个点来
        point_ind = np.argmin(self.possibility[self.split][cloud_idx])

        # Get all points within the cloud from tree structure
        # KDTree中保存了点云的 x,y,z值， N*3
        points = np.array(self.trees[self.split][cloud_idx].data, copy=False)

        # Center point of input region
        # 提取出找到的种子点的坐标
        center_point = points[point_ind, :].reshape(1, -1)
        # print(cloud_idx, center_point)

        # Add noise to the center point
        noise = np.random.normal(scale=self.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)

        # Check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < self.n_pts:
            # Query all points within the cloud   shape:(2,1,k) distance,index
            queried_idx = self.trees[self.split][cloud_idx].query(pick_point, k=len(points))[1][0]
        else:
            # Query the predefined number of points
            queried_idx = self.trees[self.split][cloud_idx].query(pick_point, k=self.n_pts)[1][0]

        # shuffle index
        idx = np.arange(len(queried_idx))
        np.random.shuffle(idx)
        queried_idx = queried_idx[idx]

        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]
        queried_pc_xyz = queried_pc_xyz - pick_point
        queried_pc_colors = self.colors[self.split][cloud_idx][queried_idx]
        queried_pc_labels = self.labels[self.split][cloud_idx][queried_idx]

        # 到此为止，先选了一个点云，然后在里面选了一个点，以这个点为中心，找到了与其最近的k个点，打乱顺序并进行中心化，得到 x y z red green blue label

        # Update the possibility of the selected points
        # 按照与选取的种子点的距离远近，来增加其possibility值，距离越近，加的越多
        dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[self.split][cloud_idx][queried_idx] += delta
        self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

        # up_sampled with replacement
        if len(points) < self.n_pts:
            queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = DataProcessing.data_aug(queried_pc_xyz,
                                                                                            queried_pc_colors,
                                                                                            queried_pc_labels,
                                                                                            queried_idx, self.n_pts)

        features = np.concatenate([queried_pc_xyz, queried_pc_colors], axis=-1)
        input_points = None
        input_neighbors = None
        input_pools = None
        input_up_samples = None

        for i in range(self.num_layers):
            # (1,N1,3)
            batch_queried_pc_xyz = queried_pc_xyz[np.newaxis, :]
            # (1,N1,k)
            batch_neighbor_idx = DataProcessing.knn_search(batch_queried_pc_xyz, batch_queried_pc_xyz, self.k_n)
            # 对输入的点云，求每个点云的k近邻点的索引 (N1,K)
            neighbour_idx = np.squeeze(batch_neighbor_idx, axis=0)

            # 对点云及其k近邻点索引进行下采样，因为前面将点云数据打乱了，所以直接去减 1/n的点等价于随机下采样
            sub_points = queried_pc_xyz[:np.shape(queried_pc_xyz)[0] // self.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:np.shape(queried_pc_xyz)[0] // self.sub_sampling_ratio[i], :]
            # 寻找初始点云在下采样点云中的最近邻点的索引
            batch_sub_points = sub_points[np.newaxis, :]
            batch_up_i = DataProcessing.knn_search(batch_sub_points, batch_queried_pc_xyz, 1)
            up_i = np.squeeze(batch_up_i, axis=0)

            if input_points is None:
                input_points = queried_pc_xyz
            else:
                input_points = np.concatenate((input_points, queried_pc_xyz), axis=0)

            if input_neighbors is None:
                input_neighbors = neighbour_idx
            else:
                input_neighbors = np.concatenate((input_neighbors, neighbour_idx), axis=0)

            if input_pools is None:
                input_pools = pool_i
            else:
                input_pools = np.concatenate((input_pools, pool_i), axis=0)

            if input_up_samples is None:
                input_up_samples = up_i
            else:
                input_up_samples = np.concatenate((input_up_samples, up_i), axis=0)

            queried_pc_xyz = sub_points

        input_points = input_points.transpose((1, 0))  # 3, N

        return torch.from_numpy(input_points), torch.from_numpy(input_neighbors), torch.from_numpy(
            input_pools), torch.from_numpy(input_up_samples), torch.from_numpy(features), torch.from_numpy(
            queried_pc_labels)
        # bat_files = self.train_files[0]
        # pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = [],[],[],[],[],[]
        # for i in bat_files:
        #     pc, sem_label, ins_label, psem_onehot_label, bbvert_padded_label, pmask_padded_label = \
        #         Data_S3DIS.load_fixed_points(i)
        #     pc_xyzrgb.append(pc)
        #     sem_labels.append(sem_label)
        #     ins_labels.append(ins_label)
        #     psem_onehot_labels.append(psem_onehot_label)
        #     bbvert_padded_labels.append(bbvert_padded_label)
        #     pmask_padded_labels.append(pmask_padded_label)
        # pc_xyzrgb = np.stack(pc_xyzrgb)
        # sem_labels = np.stack(sem_labels)
        # ins_labels = np.stack(ins_labels)
        # psem_onehot_labels = np.stack(psem_onehot_labels)
        # bbvert_padded_labels = np.stack(bbvert_padded_labels)
        # pmask_padded_labels = np.stack(pmask_padded_labels)
        #
        # flat_inputs = self.tf_map(pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels)
        #
        # num_layers = self.num_layers
        # inputs = {}
        # inputs['xyz'] = []
        # for tmp in flat_inputs[:num_layers]:
        #     inputs['xyz'].append((torch.from_numpy(tmp).float()))
        # inputs['neigh_idx'] = []
        # for tmp in flat_inputs[num_layers: 2 * num_layers]:
        #     inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        # inputs['sub_idx'] = []
        # for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
        #     inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        # inputs['interp_idx'] = []
        # for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
        #     inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        # inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float() # B,C,N
        # inputs['sem_labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        # inputs['ins_labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        # inputs['psem_onehot_labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).int()
        # inputs['bbvert_padded_labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 4]).float()
        # inputs['pmask_padded_labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 5]).float()

        # self.train_next_bat_index += 1

    def tf_map(self, pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels):
        features = pc_xyzrgb
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        batch_pc = pc_xyzrgb[:,:,:3]
        for i in range(self.num_layers):
            neighbors_idx = DataProcessing.knn_search(batch_pc, batch_pc, self.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] // self.sub_sampling_ratio[i], :]
            pool_idx = neighbors_idx[:, :batch_pc.shape[1] // self.sub_sampling_ratio[i], :]
            up_idx = DataProcessing.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbors_idx)
            input_pools.append(pool_idx)
            input_up_samples.append(up_idx)
            batch_pc = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [features, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels]

        return input_list

    def load_test_next_batch_random(self):
        idx = random.sample(range(len(self.test_files)), self.train_batch_size)
        bat_pc = []
        bat_sem_labels = []
        bat_ins_labels = []
        bat_psem_onehot_labels = []
        bat_bbvert_padded_labels = []
        bat_pmask_padded_labels = []
        for i in idx:
            file = self.test_files[i]
            pc, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = Data_S3DIS.load_fixed_points(
                file)
            bat_pc.append(pc)
            bat_sem_labels.append(sem_labels)
            bat_ins_labels.append(ins_labels)
            bat_psem_onehot_labels.append(psem_onehot_labels)
            bat_bbvert_padded_labels.append(bbvert_padded_labels)
            bat_pmask_padded_labels.append(pmask_padded_labels)

        bat_pc = np.asarray(bat_pc, dtype=np.float32)
        bat_sem_labels = np.asarray(bat_sem_labels, dtype=np.float32)
        bat_ins_labels = np.asarray(bat_ins_labels, dtype=np.float32)
        bat_psem_onehot_labels = np.asarray(bat_psem_onehot_labels, dtype=np.float32)
        bat_bbvert_padded_labels = np.asarray(bat_bbvert_padded_labels, dtype=np.float32)
        bat_pmask_padded_labels = np.asarray(bat_pmask_padded_labels, dtype=np.float32)

        return bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot_labels, bat_bbvert_padded_labels, bat_pmask_padded_labels

    def load_test_next_batch_sq_randla(self, bat_files, is_train=False):
        pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = [], [], [], [], [], []
        for i in bat_files:
            pc, sem_label, ins_label, psem_onehot_label, bbvert_padded_label, pmask_padded_label = \
                Data_S3DIS.load_fixed_points(i, is_train)
            pc_xyzrgb.append(pc)
            sem_labels.append(sem_label)
            ins_labels.append(ins_label)
            psem_onehot_labels.append(psem_onehot_label)
            bbvert_padded_labels.append(bbvert_padded_label)
            pmask_padded_labels.append(pmask_padded_label)
        pc_xyzrgb = np.stack(pc_xyzrgb)
        sem_labels = np.stack(sem_labels)
        ins_labels = np.stack(ins_labels)
        psem_onehot_labels = np.stack(psem_onehot_labels)
        bbvert_padded_labels = np.stack(bbvert_padded_labels)
        pmask_padded_labels = np.stack(pmask_padded_labels)

        flat_inputs = self.tf_map(pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels,
                                  pmask_padded_labels)

        num_layers = self.num_layers
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


    def load_test_next_batch_sq(self, bat_files):
        bat_pc = []
        bat_sem_labels = []
        bat_ins_labels = []
        bat_psem_onehot_labels = []
        bat_bbvert_padded_labels = []
        bat_pmask_padded_labels = []
        for file in bat_files:
            pc, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = Data_S3DIS.load_fixed_points(
                file)
            bat_pc += [pc]
            bat_sem_labels += [sem_labels]
            bat_ins_labels += [ins_labels]
            bat_psem_onehot_labels += [psem_onehot_labels]
            bat_bbvert_padded_labels += [bbvert_padded_labels]
            bat_pmask_padded_labels += [pmask_padded_labels]

        bat_pc = np.asarray(bat_pc, dtype=np.float32)
        bat_sem_labels = np.asarray(bat_sem_labels, dtype=np.float32)
        bat_ins_labels = np.asarray(bat_ins_labels, dtype=np.float32)
        bat_psem_onehot_labels = np.asarray(bat_psem_onehot_labels, dtype=np.float32)
        bat_bbvert_padded_labels = np.asarray(bat_bbvert_padded_labels, dtype=np.float32)
        bat_pmask_padded_labels = np.asarray(bat_pmask_padded_labels, dtype=np.float32)

        return bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot_labels, bat_bbvert_padded_labels, bat_pmask_padded_labels, bat_files

    def shuffle_train_files(self, ep):
        index = list(range(len(self.train_files)))
        random.seed(ep)
        shuffle(index)
        train_files_new = []
        for i in index:
            train_files_new.append(self.train_files[i])
        self.train_files = train_files_new
        self.train_next_bat_index = 0
        print('train files shuffled!')