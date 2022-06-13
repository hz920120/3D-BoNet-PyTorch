import glob
import numpy as np
import random
import copy
from random import shuffle
import h5py
import torch
from data_process import DataProcessing

class Data_Configs:
    sem_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
                 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
    sem_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    points_cc = 9
    sem_num = len(sem_names)
    ins_max_num = 24
    train_pts_num = 4096
    test_pts_num = 4096
    num_layers = 5
    k_n: 16
    sub_sampling_ratio: [4, 4, 4, 4, 2]


class Data_S3DIS:
    def __init__(self, dataset_path, train_areas, test_areas, train_batch_size=4):
        self.root_folder_4_traintest = dataset_path
        self.train_files = self.load_full_file_list(areas=train_areas)
        self.test_files = self.load_full_file_list(areas=test_areas)
        print('train files:', len(self.train_files))
        print('test files:', len(self.test_files))

        self.ins_max_num = Data_Configs.ins_max_num
        self.train_batch_size = train_batch_size
        self.total_train_batch_num = len(self.train_files) // self.train_batch_size

        self.num_layers = Data_Configs.num_layers
        self.k_n = Data_Configs.k_n
        self.sub_sampling_ratio = Data_Configs.sub_sampling_ratio

        self.train_next_bat_index = 0

    def load_full_file_list(self, areas):
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

        return all_files

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
    def get_bbvert_pmask_labels(pc, ins_labels):
        gt_bbvert_padded = np.zeros((Data_Configs.ins_max_num, 2, 3), dtype=np.float32)
        gt_pmask = np.zeros((Data_Configs.ins_max_num, pc.shape[0]), dtype=np.float32)
        count = -1
        unique_ins_labels = np.unique(ins_labels)
        for ins_ind in unique_ins_labels:
            if ins_ind <= -1: continue
            count += 1
            if count >= Data_Configs.ins_max_num: print('ignored! more than max instances:',
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
    def load_fixed_points(file_path):
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

        psem_onehot_labels = np.zeros((pc_xyzrgb.shape[0], Data_Configs.sem_num), dtype=np.int8)
        for idx, s in enumerate(sem_labels):
            if sem_labels[idx] == -1: continue  # invalid points
            sem_idx = Data_Configs.sem_ids.index(s)
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
        bat_files = self.train_files[self.train_next_bat_index *
                                     self.train_batch_size:(self.train_next_bat_index + 1) * self.train_batch_size]
        pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = [],[],[],[],[],[]
        for i in bat_files:
            pc, sem_label, ins_label, psem_onehot_label, bbvert_padded_label, pmask_padded_label = \
                Data_S3DIS.load_fixed_points(i)
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

        flat_inputs = self.tf_map(pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels)

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
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float() # B,C,N
        inputs['sem_labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['ins_labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['psem_onehot_labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).int()
        inputs['bbvert_padded_labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 4]).float()
        inputs['pmask_padded_labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 5]).float()

        self.train_next_bat_index += 1
        return inputs

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