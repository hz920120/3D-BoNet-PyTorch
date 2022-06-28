import glob
import os
import sys
from os.path import join

import h5py
import numpy as np


# parser = argparse.ArgumentParser()
# parser.add_argument('--root', help='path to root directory')
# parser.add_argument('--seed', type=int, default=42, help='random number seed')
# args = parser.parse_args()
#
# root = args.root
# seed = args.seed
# np.random.seed(seed)
# fname = os.path.join(root, 'metadata', 'classes.txt')
# classes = [line.strip() for line in open(fname, 'r')]
#
# fname = os.path.join(root, 'metadata', 'all_data.txt')
# flist = [os.path.join(root, 'processed', line.strip())
#          for line in open(fname, 'r')]
import pandas as pd
from data_process import DataProcessing as DP



def sample_cloud(cloud, num_samples):
    n = cloud.shape[0]
    if n >= num_samples:
        indices = np.random.choice(n, num_samples, replace=False)
    else:
        indices = np.random.choice(n, num_samples - n, replace=True)
        indices = list(range(n)) + list(indices)
    sampled = cloud[indices, :]
    return sampled


def room_to_blocks(cloud, num_points, size=1.0, stride=0.5, threshold=100):
    cloud[:, 3:6] /= 255.0
    limit = np.amax(cloud[:, 0:3], axis=0)
    width = int(np.ceil((limit[0] - size) / stride)) + 1
    depth = int(np.ceil((limit[1] - size) / stride)) + 1
    cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]
    blocks = []
    for (x, y) in cells:
        xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
        ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
        cond  = xcond & ycond
        if np.sum(cond) < threshold:
            continue
        block = cloud[cond, :]
        block = sample_cloud(block, num_points)
        blocks.append(block)
    blocks = np.stack(blocks, axis=0)
    # A batch should have shape of BxNx14, where
    # [0:3] - global coordinates
    # [3:6] - block normalized coordinates (centered at Z-axis)
    # [6:9] - RGB colors
    # [9:12] - room normalized coordinates
    # [12:14] - semantic and instance labels
    num_blocks = blocks.shape[0]
    batch = np.zeros((num_blocks, num_points, 14))
    for b in range(num_blocks):
        minx = min(blocks[b, :, 0])
        miny = min(blocks[b, :, 1])
        batch[b, :, 3]  = blocks[b, :, 0] - (minx + size * 0.5)
        batch[b, :, 4]  = blocks[b, :, 1] - (miny + size * 0.5)
        batch[b, :, 9]  = blocks[b, :, 0] / limit[0]
        batch[b, :, 10] = blocks[b, :, 1] / limit[1]
        batch[b, :, 11] = blocks[b, :, 2] / limit[2]
    batch[:,:, 0:3] = blocks[:,:,0:3]
    batch[:,:, 5:9] = blocks[:,:,2:6]
    batch[:,:, 12:] = blocks[:,:,6:8]
    return batch


def save_batch_h5(fname, batch):
    fp = h5py.File(fname)
    coords = batch[:, :, 0:3]
    points = batch[:, :, 3:12]
    labels = batch[:, :, 12:14]
    fp.create_dataset('coords', data=coords, compression='gzip', dtype='float32')
    fp.create_dataset('points', data=points, compression='gzip', dtype='float32')
    fp.create_dataset('labels', data=labels, compression='gzip', dtype='int64')
    fp.close()

dataset_path = '/media/cesc/CESC/data'
gt_class = [x.rstrip() for x in open(join(dataset_path, 'meta/class_names.txt'))]
anno_paths = [line.rstrip() for line in open(join(dataset_path, 'meta/anno_paths.txt'))]
anno_paths = [join(dataset_path, p) for p in anno_paths]
gt_class2label = {cls: i for i, cls in enumerate(gt_class)}
sub_grid_size = 0.02
num_points = 10240
if __name__ == '__main__':

    for annotation_path in anno_paths:
        print(annotation_path)
        elements = str(annotation_path).split('/')
        ins_idx = 1
        data_list = []
        for f in glob.glob(join(annotation_path, '*.txt')):
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

        xyz = pc_label[:, :3].astype(np.float32)
        colors = pc_label[:, 3:6].astype(np.uint8)
        labels = pc_label[:, 6].astype(np.uint8)
        ins_labels = pc_label[:, 7].astype(np.uint8)

        sub_data_list = []
        sub_xyz, sub_colors, sub_labels, sub_ins_labels = DP.grid_sub_sampling(xyz, colors, labels, ins_labels,
                                                                               sub_grid_size)


        cloud = np.concatenate([sub_xyz, sub_colors, sub_labels, sub_ins_labels], 1)



        batch = room_to_blocks(cloud, num_points, size=2.0, stride=1.5)
        elements = str(annotation_path).split('/')
        scene_name = elements[-3] + '_' + elements[-2]
        fname = os.path.join(dataset_path, 'h5', scene_name + '.h5')
        print('> Saving batch to {}...'.format(fname))
        if not os.path.exists(fname):
            save_batch_h5(fname, batch)