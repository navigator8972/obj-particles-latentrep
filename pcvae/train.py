
'''
PointCloudVariationalAutoEncoder
'''

from __future__ import print_function
import torch
import torch.utils.data
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np


from torch import nn, optim
from torch.autograd import Variable
from pcvae_dynamic import PointCloudVariationalAutoEncoderDisplacement

from pointnet import PointNetfeat, feature_transform_reguliarzer
from pcn_decoder import PCNDecoder
from ChamferDistancePyTorch import chamfer_distance_with_batch
from utils import batch_rodriguez_formula_elementwise
from visdom_utils import VisdomInterface
from loadH5Data import PointCloudDataSetFromH5_3435_seq_interpolation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    model = PointCloudVariationalAutoEncoderDisplacement(n_points=3435, latent_size=32, use_pcn=0, mode=False)
    if torch.cuda.is_available():
        model.cuda()

    # dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_dense_3435_bag72_a0_speed1.h5', mode='extrapolation')
    # print('Training -- Number of data:{}'.format(len(dataset)))
    # valid_dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_dense_3435_bag72_a0_speed1.h5', train='valid',mode='extrapolation')
    # print('Validation -- Number of data:{}'.format(len(valid_dataset)))
    # test_dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_dense_3435_bag72_a0_speed1.h5', train='test',mode='extrapolation')
    # print('Testing -- Number of data:{}'.format(len(test_dataset)))


    #dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_dense_3435_bag72_a0_speed1.h5', mode='interpolation')
    #print('Training -- Number of data:{}'.format(len(dataset)))
    #valid_dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_dense_3435_bag72_a0_speed1.h5', train='valid',mode='interpolation')
    #print('Validation -- Number of data:{}'.format(len(valid_dataset)))
    #test_dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_dense_3435_bag72_a0_speed1.h5', train='test',mode='interpolation')
    #print('Testing -- Number of data:{}'.format(len(test_dataset)))


    # dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_resample_bag72_a0_speed1_20200225.h5', mode='extrapolation')
    # print('Training -- Number of data:{}'.format(len(dataset)))
    # valid_dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_resample_bag72_a0_speed1_20200225.h5', train='valid',mode='extrapolation')
    # print('Validation -- Number of data:{}'.format(len(valid_dataset)))
    # test_dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_resample_bag72_a0_speed1_20200225.h5', train='test',mode='extrapolation')
    # print('Testing -- Number of data:{}'.format(len(test_dataset)))

    dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_dense_3435_bag72_a0_speed1_20200225.h5', mode='unseen2')
    print('Training -- Number of data:{}'.format(len(dataset)))
    valid_dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_dense_3435_bag72_a0_speed1_20200225.h5', train='valid',mode='unseen2')
    print('Validation -- Number of data:{}'.format(len(valid_dataset)))
    test_dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_dense_3435_bag72_a0_speed1_20200225.h5', train='test',mode='unseen2')
    print('Testing -- Number of data:{}'.format(len(test_dataset)))

    # model.fit(dataset, valid_dataset, batch_size=16, n_epoch=100, lr=3e-4, denoising=0.002, save_every=20, visdom=False)
    model.fit(dataset, valid_dataset, batch_size=16, n_epoch=100, lr=3e-4, denoising=0.002, save_every=10, visdom=False)

