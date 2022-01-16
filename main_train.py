import os
import torch

gpu_num = 0

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
    ]
    return devices if devices else [torch.device('cpu')]

# x = torch.ones(2, 3, device=try_gpu())
# print(x)

if __name__=='__main__':

    # from main_3D_BoNet import BoNet
    from helper_data_s3dis import Data_Configs as Data_Configs

    configs = Data_Configs()
    # net = BoNet(configs = configs)
    # net.creat_folders(name='log', re_train=False)
    # net.build_graph()

    ####
    from helper_data_s3dis import Data_S3DIS as Data
    train_areas =['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
    test_areas =['Area_5']

    dataset_path = './data_s3dis/'
    data = Data(dataset_path, train_areas, test_areas, train_batch_size=4)
    print(1)
    # train(net, data)

