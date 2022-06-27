from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob, pickle

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
import utils.helper_ply as wp
from data_process import DataProcessing as DP

dataset_path = '/media/cesc/CESC/data'
anno_paths = [line.rstrip() for line in open('/media/cesc/CESC/data/meta/anno_paths.txt')]
anno_paths = [join(dataset_path, p) for p in anno_paths]

gt_class = [x.rstrip() for x in open(join('/media/cesc/CESC/data/meta/class_names.txt'))]
gt_class2label = {cls: i for i, cls in enumerate(gt_class)}

sub_grid_size = 0.04
original_pc_folder = join(dirname(dataset_path), 'original_ply')
sub_pc_folder = join(dirname(dataset_path), 'input_{:.3f}'.format(sub_grid_size))
os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None
out_format = '.ply'


def convert_pc2ply(anno_path, save_path, scene_name):
    """
    Convert original dataset files to ply file (each line is XYZRGBL).
    We aggregated all the points from each instance in the room.
    :param anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
    :param save_path: path to save original point clouds (each line is XYZRGBL)
    :return: None
    """
    data_list = []
    ins_idx = 1
    for f in glob.glob(join(anno_path, '*.txt')):
        class_name = os.path.basename(f).split('_')[0]
        if class_name not in gt_class:  # note: in some room there is 'staris' class..
            class_name = 'clutter'
        pc = pd.read_csv(f, header=None, delim_whitespace=True).values
        labels = np.ones((pc.shape[0], 1)) * gt_class2label[class_name]
        ins_labels = np.ones((pc.shape[0], 1)) * ins_idx
        ins_idx += 1
        data_list.append(np.concatenate([pc, labels, ins_labels], 1))  # Nx7

    pc_label = np.concatenate(data_list, 0)
    xyz_min = np.amin(pc_label, axis=0)[0:3]
    pc_label[:, 0:3] -= xyz_min

    blocks = room_to_blocks(pc_label, size=2, stride=1.5, scene_name=scene_name)
    os.mkdir(join(dataset_path, 'sample_ori_dir')) if not exists(join(dataset_path, 'sample_ori_dir')) else None
    h5_path = join(dataset_path, 'sample_ori_dir', scene_name + '.h5')
    save_dic(h5_path, blocks)
    sub_save_path = join(dataset_path, 'sample_ori_dir', 'input_{:.3f}'.format(sub_grid_size))
    limit = np.amax(pc_label[:, 0:3], axis=0)
    save_sub(sub_save_path, blocks, limit)


def save_sub(sub_save_path, blocks, limit):
    for name, block in blocks.items():
        xyz = block[:, :3].astype(np.float32)
        colors = block[:, 3:6].astype(np.float32)
        labels = block[:, 9].astype(np.uint8)
        ins_labels = block[:, 10].astype(np.uint8)
        sub_xyz, sub_colors, sub_labels, sub_ins_labels = DP.grid_sub_sampling(xyz, colors, labels, ins_labels,
                                                                               sub_grid_size)
        sub_colors /= 255.0
        # limit_x = np.array(limit[0])
        # limit_y = np.array(limit[1])
        # limit_z = np.array(limit[2])

        os.mkdir(sub_save_path) if not exists(sub_save_path) else None
        sub_ply_file = join(sub_save_path, name + '.ply')
        wp.write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels, sub_ins_labels],['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'ins_labels'])

        search_tree = KDTree(sub_xyz)
        kd_tree_file = join(sub_save_path, name + '_KDTree.pkl')
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)

        proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
        proj_idx = proj_idx.astype(np.int32)
        proj_save = join(sub_save_path, name + '_proj.pkl')
        with open(proj_save, 'wb') as f:
            pickle.dump([proj_idx, labels], f)

def room_to_blocks(cloud, size=1.0, stride=0.5, threshold=100, scene_name=None):
    if scene_name is None:
        print('wrong scene!')
        return None
    # cloud[:, 3:6] /= 255.0
    limit = np.amax(cloud[:, 0:3], axis=0)
    width = int(np.ceil((limit[0] - size) / stride)) + 1
    depth = int(np.ceil((limit[1] - size) / stride)) + 1
    cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]
    blocks = {}
    i = 0
    for (x, y) in cells:
        xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
        ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
        cond  = xcond & ycond
        if np.sum(cond) < threshold:
            continue
        block = cloud[cond, :]
        block_temp = np.zeros((block.shape[0], 11))
        # print(block.shape)
        # [0:3] - global coordinates
        # [3:6] - RGB colors
        # [6:9] - room normalized coordinates
        # [9:11] - semantic and instance labels
        block_temp[:, 0:6] = block[:, 0:6]
        block_temp[:, 6] = block[:, 0] / limit[0]
        block_temp[:, 7] = block[:, 1] / limit[1]
        block_temp[:, 8] = block[:, 2] / limit[2]
        block_temp[:, 9:11] = block[:, 6:8]
        blocks[scene_name + '_' + str(i).zfill(4)] = (block_temp)
        i += 1
    return blocks

def save_dic(name, batch):
    file = open(name, "wb")
    pickle.dump(batch, file)
    file.close()
    # a_file = open("data.pkl", "rb")
    # output = pickle.load(a_file)
    # print(output)
    # a_file.close()

class FError(Exception):
    pass

if __name__ == '__main__':
    # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
    for annotation_path in anno_paths:
        print(annotation_path)
        elements = str(annotation_path).split('/')
        out_file_name = elements[-3] + '_' + elements[-2] + out_format
        scene_name = elements[-3] + '_' + elements[-2]
        convert_pc2ply(annotation_path, join(original_pc_folder, out_file_name), scene_name)