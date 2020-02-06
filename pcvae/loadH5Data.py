
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
from pcvae import PointCloudVariationalAutoEncoder

from pointnet import PointNetfeat, feature_transform_reguliarzer
from pcn_decoder import PCNDecoder
from ChamferDistancePyTorch import chamfer_distance_with_batch
from utils import batch_rodriguez_formula_elementwise
from visdom_utils import VisdomInterface

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PointCloudDataSetFromH5(Dataset):
    def __init__(self, h5_file, norm=True, train=True, float_type=np.float32, rand_seed=1234):
        if isinstance(train, bool):
            train = 'train' if train else 'test'

        # full_data = np.load(npz_file)['data'].astype(float_type)
        full_data = h5py.File(h5_file, "r")
        numVideo = full_data['posSeq'].shape[0]
        numFrame = full_data['posSeq'].shape[1]
        numPoint = full_data['posSeq'].shape[2]
        numTotalFrame = numVideo*numFrame

        h5data_seqpos = full_data['posSeq'][:].reshape(numVideo*numFrame,numPoint,3).transpose(0,2,1).astype(float_type)
        # h5data_actionlabel = full_data['actionid'][:].reshape(numVideo*numFrame,numPoint,1).astype(int)


        # shuffle the data
        np.random.seed(rand_seed)
        indices = np.arange(numTotalFrame)
        np.random.shuffle(indices)
        h5data_seqpos = h5data_seqpos[indices]
        # h5data_actionlabel = h5data_actionlabel[indices]

        if train == 'train':
            self.pc_data = h5data_seqpos[:int(h5data_seqpos.shape[0] * .8)]
            self.pc_valid = h5data_seqpos[:int(h5data_seqpos.shape[0] * .8)]
            # self.pc_label = h5data_actionlabel[:int(h5data_seqpos.shape[0] * .8)]
        elif train == 'test':
            self.pc_data = h5data_seqpos[int(h5data_seqpos.shape[0] * .9):]
            # self.pc_label = h5data_actionlabel[int(h5data_seqpos.shape[0] * .9):]
        else:
            self.pc_data = h5data_seqpos[int(h5data_seqpos.shape[0] * .8):int(h5data_seqpos.shape[0] * .9)]  # for validation
            # self.pc_label = h5data_actionlabel[int(h5data_seqpos.shape[0] * .8):int(h5data_seqpos.shape[0] * .9)]

        self.pc_label = None
        return

    def __len__(self):
        return len(self.pc_data)

    def __getitem__(self, idx):
        if self.pc_label is None:
            return self.pc_data[idx]
        else:
            return self.pc_data[idx], self.pc_label[idx]

if __name__ == '__main__':
    pcdataset = PointCloudDataSetFromH5('../data/bagdata.h5')
    print(pcdataset.pc_data.shape)
    # model = PointCloudVariationalAutoEncoder(n_points=1145, latent_size=32, use_pcn=2)
    # if torch.cuda.is_available():
    #     model.cuda()
    #
    # dataset = PointCloudDataSetFromH5('../data/bagdata.h5')
    # print('Training -- Number of data:{}'.format(len(dataset)))
    # valid_dataset = PointCloudDataSetFromH5('../data/bagdata.h5', train='valid')
    #
    # print('Validation -- Number of data:{}'.format(len(valid_dataset)))
    #
    # model.fit(dataset, valid_dataset, batch_size=32, n_epoch=3000, lr=3e-4, denoising=0.002, save_every=100, visdom=False)
