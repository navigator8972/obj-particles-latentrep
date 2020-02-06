import h5py
from __future__ import print_function
from torch.utils.data import Dataset, DataLoader
import numpy as np

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
