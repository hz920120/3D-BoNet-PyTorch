import glob
from torch.utils.tensorboard import SummaryWriter
import h5py
import numpy as np

areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
path = './Data_S3DIS/'


def load_raw_data_file_s3dis_block(file_path):
    # block_id = int(file_path[-4:])
    # file_path = file_path[0:-5]

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
if __name__ == '__main__':
    summary_path = './logs_count'
    writer = SummaryWriter(summary_path)
    block = 0
    total_ins = 0
    total_sem = 0
    sem_max = 0
    ins_max = 0
    for a in areas:
        print('check area:', a)
        files = sorted(glob.glob(path + a + '*.h5'))

        for f in files:
            fin = h5py.File(f, 'r')
            coords = fin['coords'][:]

            shape = len(coords)
            for block_id in range(shape):
                coords = fin['coords'][block_id]
                points = fin['points'][block_id]
                semIns_labels = fin['labels'][block_id]
                sem_labels = semIns_labels[:, 0]
                ins_labels = semIns_labels[:, 1]
                count_sem_num_per_block = np.unique(sem_labels)
                count_ins_num_per_block = np.unique(ins_labels)

                sem_max = len(count_sem_num_per_block) if len(count_sem_num_per_block) > sem_max else sem_max
                ins_max = len(count_ins_num_per_block) if len(count_ins_num_per_block) > ins_max else ins_max
                total_sem += len(count_sem_num_per_block)
                total_ins += len(count_ins_num_per_block)
                writer.add_scalar('sem_num_per_block', len(count_sem_num_per_block), block)
                writer.add_scalar('ins_num_per_block', len(count_ins_num_per_block), block)
                block += 1
            print('check {} ends'.format(f))

    print('avg ins : {}'.format(total_ins / block))
    print('avg sem : {}'.format(total_sem / block))
    print('ins max : {}'.format(ins_max))
    print('sem max : {}'.format(sem_max))


#4096
# avg ins : 5.627502539891233
# avg sem : 4.360485268630849
# ins max : 21
# sem max : 9


#20480
# avg ins : 9.96359017781541
# avg sem : 5.770956816257409
# ins max : 27
# sem max : 11