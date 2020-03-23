
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PointCloudDataSetFromH5(Dataset):
    def __init__(self, h5_file, norm=True, train=True, float_type=np.float32, rand_seed=1234, denseVer = 5):
        if isinstance(train, bool):
            train = 'train' if train else 'test'

        self.pcl_key = "posSeqDense_{}".format(denseVer)
        # full_data = np.load(npz_file)['data'].astype(float_type)
        self.full_data = h5py.File(h5_file, "r")
        self.numVideo = self.full_data[self.pcl_key].shape[0]
        self.numFrame = self.full_data[self.pcl_key].shape[1]
        self.numPoint = self.full_data[self.pcl_key].shape[2]
        self.numTotalFrame = self.numVideo*self.numFrame
        self.float_type = float_type

        # h5data_seqpos = full_data['posSeqDense_5'][:].reshape(numVideo*numFrame,numPoint,3).transpose(0,2,1).astype(float_type)


        # shuffle the data
        np.random.seed(rand_seed)
        self.indices = np.arange(self.numTotalFrame)
        np.random.shuffle(self.indices)

        # h5data_seqpos = h5data_seqpos[indices]
        #
        if train == 'train':
            # self.pc_data = h5data_seqpos[:int(h5data_seqpos.shape[0] * .8)]
            # self.pc_valid = h5data_seqpos[:int(h5data_seqpos.shape[0] * .8)]
            self.pc_data_index = self.indices[:int(self.numTotalFrame * .8)]
            self.pc_valid_index = self.indices[:int(self.numTotalFrame * .8)]
        elif train == 'test':
            # self.pc_data = h5data_seqpos[int(numTotalFrame * .9):]
            self.pc_data_index = self.indices[int(self.numTotalFrame * .9):]
        else:
            # self.pc_data = h5data_seqpos[int(numTotalFrame * .8):int(numTotalFrame * .9)]  # for validation
            self.pc_data_index = self.indices[int(self.numTotalFrame * .8):int(self.numTotalFrame * .9)]  # for validation

        self.pc_label_index = None
        return

    def __len__(self):
        return len(self.pc_data_index)

    def __getitem__(self, idx):
        sampleindex_seq = self.pc_data_index[idx]//self.numFrame
        # sampleindex_frame = self.pc_data_index[idx]%self.numFrame
        sampleindex_frame = self.pc_data_index[idx]%self.numFrame

        if self.pc_label_index is None:
            pcl_data = self.full_data[self.pcl_key][sampleindex_seq, sampleindex_frame, :].reshape(self.numPoint,
                                                                                                   3).T.astype(
                self.float_type)
            # for i in range(len(sampleindex_seq)):
            #     pcl_data = self.full_data[self.pcl_key][sampleindex_seq,sampleindex_frame,:].reshape(self.numPoint, 3).T.astype(
            #     self.float_type)
            return pcl_data
        else:
            pcl_data = self.full_data[self.pcl_key][sampleindex_seq, sampleindex_frame, :].reshape(self.numPoint,
                                                                                                   3).T.astype(
                self.float_type)
            return pcl_data, None

class PointCloudDataSetFromH5_3435_seq(Dataset):
    def __init__(self, h5_file, norm=True, train=True, float_type=np.float32, rand_seed=1234):
        if isinstance(train, bool):
            train = 'train' if train else 'test'

        self.pcl_key = "posSeqDense_{}".format("3435")
        # full_data = np.load(npz_file)['data'].astype(float_type)
        self.full_data = h5py.File(h5_file, "r")
        self.numVideo = self.full_data[self.pcl_key].shape[0]
        self.numFrame = self.full_data[self.pcl_key].shape[1]
        self.numPoint = self.full_data[self.pcl_key].shape[2]
        self.numTotalFrame = self.numVideo*self.numFrame
        self.float_type = float_type

        # shuffle the data
        np.random.seed(rand_seed)
        self.indices = np.arange(self.numVideo)
        np.random.shuffle(self.indices)

        # h5data_seqpos = h5data_seqpos[indices]
        #
        if train == 'train':
            self.pc_data_index = self.indices[:int(self.numVideo * .8)]
            self.pc_valid_index = self.indices[:int(self.numVideo * .8)]
        elif train == 'test':
            self.pc_data_index = self.indices[int(self.numVideo * .9):]
        else:
            self.pc_data_index = self.indices[int(self.numVideo * .8):int(self.numVideo * .9)]  # for validation

        self.pc_label_index = None
        return

    def __len__(self):
        return len(self.pc_data_index)

    def __getitem__(self, idx):
        sampleindex_seq = self.pc_data_index[idx]
        if self.pc_label_index is None:
            pcl_data = self.full_data[self.pcl_key][sampleindex_seq, :].transpose(0,2,1).astype(
                self.float_type)
            return pcl_data
        else:
            pcl_data = self.full_data[self.pcl_key][sampleindex_seq, :].transpose(0,2,1).astype(
                self.float_type)
            return pcl_data, None

class PointCloudDataSetFromH5_3435_seq_interpolation_pre(Dataset):
    def __init__(self, h5_file, norm=True, train=True, float_type=np.float32, rand_seed=1234, mode='interpolation', preaction = 0):
        if isinstance(train, bool):
            train = 'train' if train else 'test'

        # self.pcl_key = "posSeqDense_{}".format("3435")
        self.pcl_key = "posSeqDense_3435"
        # self.pcl_key = "posSeq"
        # full_data = np.load(npz_file)['data'].astype(float_type)
        self.full_data = h5py.File(h5_file, "r")

        # get the valid record
        if preaction is not None:
            validVideoInd = []
            for i in range(self.full_data[self.pcl_key].shape[0]):
                if self.full_data['initActionStep'][i] == preaction:
                    validVideoInd.append(i)

            print(i)

        self.numVideo = self.full_data[self.pcl_key].shape[0]
        self.numFrame = self.full_data[self.pcl_key].shape[1]
        self.numPoint = self.full_data[self.pcl_key].shape[2]
        self.numTotalFrame = self.numVideo*self.numFrame
        self.float_type = float_type
        self.mode = mode # mode of testing/validation
        self.numValidFrame = self.numFrame - 2


        self.lefthandind = [363,558,754,755,757,758,759,760,761,762,763,764,765,766,767,768,769,770,771,780,781,782,785,913,914,915,918]
        self.righthandind = [452,787,788,789,792,793,794,878,893,894,895,910,911,912,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1041,1042]


        # shuffle the data
        np.random.seed(rand_seed)
        if mode == 'randomSelect':
            self.indices = np.arange(self.numTotalFrame)
            # don't use the 1st and last frame because the acceleration is not 0

            great_ind = []
            for i in range(len(self.indices)):
                if (self.indices[i] % self.numFrame != 0) and (self.indices[i] % self.numFrame != (self.numFrame - 1)):
                    great_ind.append(i)

            self.indices = self.indices[great_ind]

            np.random.shuffle(self.indices)

            if train == 'train':
                self.pc_data_index = self.indices[:int(self.numTotalFrame * .8)]
                self.pc_valid_index = self.indices[:int(self.numTotalFrame * .8)]
            elif train == 'test':
                self.pc_data_index = self.indices[int(self.numTotalFrame * .9):]
            else:
                self.pc_data_index = self.indices[
                                     int(self.numTotalFrame * .8):int(self.numTotalFrame * .9)]  # for validation
        elif mode == 'unseen1':
            # unseen, one frame estimation
            self.indices = np.arange(self.numTotalFrame)
            # self.trainvideonum = int(self.numVideo * 0.8)
            # self.valvideonum = int(self.numVideo * 0.1)
            # self.tstvideonum = int(self.numVideo * 0.1)

            # print(self.trainvideonum)
            # print(self.valvideonum)
            # print(self.tstvideonum)


            if train == 'train':
                self.pc_data_index = self.indices[:int(self.numVideo * 0.8)*self.numFrame]
                self.pc_valid_index = self.indices[:int(self.numVideo * 0.8)*self.numFrame]
            elif train == 'test':
                self.pc_data_index = self.indices[int(self.numVideo * 0.9)*self.numFrame:]
            else:
                self.pc_data_index = self.indices[
                                     int(self.numVideo * 0.8)*self.numFrame:int(self.numVideo * 0.9)*self.numFrame]  # for validation


            # print(self.pc_data_index)
            great_ind = []
            for i in range(len(self.pc_data_index)):
                if (self.pc_data_index[i] % self.numFrame != 0) and (self.pc_data_index[i] % self.numFrame != (self.numFrame - 1)):
                    great_ind.append(i)

            self.pc_data_index = self.pc_data_index[great_ind]

            np.random.shuffle(self.pc_data_index)
        elif mode == 'unseen2':
            # long term estimation
            # only do shuffling in the training set
            self.indices = np.arange(self.numTotalFrame)
            if train == 'train':
                self.pc_data_index = self.indices[:int(self.numVideo * 0.8)*self.numFrame]
                self.pc_valid_index = self.indices[:int(self.numVideo * 0.8)*self.numFrame]
                np.random.shuffle(self.pc_data_index)
            elif train == 'test':
                self.pc_data_index = self.indices[int(self.numVideo * 0.9)*self.numFrame:]
            else:
                self.pc_data_index = self.indices[
                                     int(self.numVideo * 0.8)*self.numFrame:int(self.numVideo * 0.9)*self.numFrame]  # for validation
                
            great_ind = []
            for i in range(len(self.pc_data_index)):
                if (self.pc_data_index[i] % self.numFrame != 0) and (self.pc_data_index[i] % self.numFrame != (self.numFrame - 1)):
                    great_ind.append(i)

            self.pc_data_index = self.pc_data_index[great_ind]
            

            
        else:
            self.indexpervideo = np.arange(1, self.numFrame - 1)  # 1->38
            if mode == 'interpolation':
                np.random.shuffle(self.indexpervideo)
            elif mode == 'extrapolation':
                pass
            numTrain = 18
            numVal = 10
            numTst = 10

            # indices of train,val,tst per sequence
            self.trainindices = self.indexpervideo[:numTrain]
            self.valindices = self.indexpervideo[numTrain:numTrain+numVal]
            self.tstindices = self.indexpervideo[numTrain+numVal:numTrain+numVal+numTst]

            # initialization
            self.trainindices_total = np.copy(self.trainindices)
            self.valindices_total = np.copy(self.valindices)
            self.tstindices_total = np.copy(self.tstindices)

            for i in range(1, self.numVideo):
                self.trainindices_total = np.hstack((self.trainindices_total, self.trainindices+self.numFrame*i))
                self.valindices_total = np.hstack((self.valindices_total, self.valindices+self.numFrame*i))
                self.tstindices_total = np.hstack((self.tstindices_total, self.tstindices+self.numFrame*i))

            np.random.shuffle(self.trainindices_total)
            np.random.shuffle(self.valindices_total)
            np.random.shuffle(self.tstindices_total)

            # h5data_seqpos = h5data_seqpos[indices]
            #
            if train == 'train':
                self.pc_data_index = self.trainindices_total
                self.pc_valid_index = self.trainindices_total
            elif train == 'test':
                self.pc_data_index = self.tstindices_total
            else:
                self.pc_data_index = self.valindices_total




        self.pc_label_index = None
        return

    def __len__(self):
        return len(self.pc_data_index)

    def __getitem__(self, idx):
        sampleindex_seq = self.pc_data_index[idx]//self.numFrame
        sampleindex_frame = self.pc_data_index[idx]%self.numFrame

        if self.pc_label_index is None:
            pcl_data = self.full_data[self.pcl_key][sampleindex_seq, sampleindex_frame, :].reshape(self.numPoint,3).T.astype(self.float_type)
            pcl_data_nextframe = self.full_data[self.pcl_key][sampleindex_seq, sampleindex_frame + 1, :].reshape(self.numPoint,3).T.astype(self.float_type)

            return pcl_data, pcl_data_nextframe
        else:
            pcl_data = self.full_data[self.pcl_key][sampleindex_seq, sampleindex_frame, :].reshape(self.numPoint,3).T.astype(self.float_type)
            pcl_data_nextframe = self.full_data[self.pcl_key][sampleindex_seq, sampleindex_frame + 1, :].reshape(self.numPoint,3).T.astype(self.float_type)

            return pcl_data, pcl_data_nextframe

class PointCloudDataSetFromH5_3435_seq_interpolation(Dataset):
    def __init__(self, h5_file, norm=True, train=True, float_type=np.float32, rand_seed=1234, mode='interpolation', preaction = 0):
        if isinstance(train, bool):
            train = 'train' if train else 'test'

        # self.pcl_key = "posSeqDense_{}".format("3435")
        self.pcl_key = "posSeqDense_3435"
        # self.pcl_key = "posSeq"
        # full_data = np.load(npz_file)['data'].astype(float_type)
        self.full_data = h5py.File(h5_file, "r")

        # get the valid record
        self.validVideoInd = []
        for i in range(self.full_data[self.pcl_key].shape[0]):
            if (preaction is not None) and (self.full_data['initactionid'][i] == preaction):
                self.validVideoInd.append(i)
            elif preaction is None:
                self.validVideoInd.append(i)

        self.numVideo = len(self.validVideoInd)
        self.numFrame = self.full_data[self.pcl_key].shape[1]
        self.numPoint = self.full_data[self.pcl_key].shape[2]

        self.numTotalFrame = self.numVideo*self.numFrame
        self.float_type = float_type
        self.mode = mode # mode of testing/validation
        self.numValidFrame = self.numFrame - 2


        self.lefthandind = [363,558,754,755,757,758,759,760,761,762,763,764,765,766,767,768,769,770,771,780,781,782,785,913,914,915,918]
        self.righthandind = [452,787,788,789,792,793,794,878,893,894,895,910,911,912,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1041,1042]


        # shuffle the data
        np.random.seed(rand_seed)
        if mode == 'randomSelect':
            self.indices = np.arange(self.numTotalFrame)
            # don't use the 1st and last frame because the acceleration is not 0

            great_ind = []
            for i in range(len(self.indices)):
                if (self.indices[i] % self.numFrame != 0) and (self.indices[i] % self.numFrame != (self.numFrame - 1)):
                    great_ind.append(i)

            self.indices = self.indices[great_ind]

            np.random.shuffle(self.indices)

            if train == 'train':
                self.pc_data_index = self.indices[:int(self.numTotalFrame * .8)]
                self.pc_valid_index = self.indices[:int(self.numTotalFrame * .8)]
            elif train == 'test':
                self.pc_data_index = self.indices[int(self.numTotalFrame * .9):]
            else:
                self.pc_data_index = self.indices[
                                     int(self.numTotalFrame * .8):int(self.numTotalFrame * .9)]  # for validation
        elif mode == 'unseen1':
            # unseen, one frame estimation
            self.indices = np.arange(self.numTotalFrame)
            # self.trainvideonum = int(self.numVideo * 0.8)
            # self.valvideonum = int(self.numVideo * 0.1)
            # self.tstvideonum = int(self.numVideo * 0.1)

            # print(self.trainvideonum)
            # print(self.valvideonum)
            # print(self.tstvideonum)


            if train == 'train':
                self.pc_data_index = self.indices[:int(self.numVideo * 0.8)*self.numFrame]
                self.pc_valid_index = self.indices[:int(self.numVideo * 0.8)*self.numFrame]
            elif train == 'test':
                self.pc_data_index = self.indices[int(self.numVideo * 0.9)*self.numFrame:]
            else:
                self.pc_data_index = self.indices[
                                     int(self.numVideo * 0.8)*self.numFrame:int(self.numVideo * 0.9)*self.numFrame]  # for validation


            # print(self.pc_data_index)
            great_ind = []
            for i in range(len(self.pc_data_index)):
                if (self.pc_data_index[i] % self.numFrame != 0) and (self.pc_data_index[i] % self.numFrame != (self.numFrame - 1)):
                    great_ind.append(i)

            self.pc_data_index = self.pc_data_index[great_ind]

            np.random.shuffle(self.pc_data_index)
        elif mode == 'unseen2':
            # long term estimation
            # only do shuffling in the training set
            self.indices = np.arange(self.numTotalFrame)
            if train == 'train':
                self.pc_data_index = self.indices[:int(self.numVideo * 0.8)*self.numFrame]
                self.pc_valid_index = self.indices[:int(self.numVideo * 0.8)*self.numFrame]
                np.random.shuffle(self.pc_data_index)
            elif train == 'test':
                self.pc_data_index = self.indices[int(self.numVideo * 0.9)*self.numFrame:]
            else:
                self.pc_data_index = self.indices[
                                     int(self.numVideo * 0.8)*self.numFrame:int(self.numVideo * 0.9)*self.numFrame]  # for validation
                
            great_ind = []
            for i in range(len(self.pc_data_index)):
                if (self.pc_data_index[i] % self.numFrame != 0) and (self.pc_data_index[i] % self.numFrame != (self.numFrame - 1)):
                    great_ind.append(i)

            self.pc_data_index = self.pc_data_index[great_ind]
            

            
        else:
            self.indexpervideo = np.arange(1, self.numFrame - 1)  # 1->38
            if mode == 'interpolation':
                np.random.shuffle(self.indexpervideo)
            elif mode == 'extrapolation':
                pass
            numTrain = 18
            numVal = 10
            numTst = 10

            # indices of train,val,tst per sequence
            self.trainindices = self.indexpervideo[:numTrain]
            self.valindices = self.indexpervideo[numTrain:numTrain+numVal]
            self.tstindices = self.indexpervideo[numTrain+numVal:numTrain+numVal+numTst]

            # initialization
            self.trainindices_total = np.copy(self.trainindices)
            self.valindices_total = np.copy(self.valindices)
            self.tstindices_total = np.copy(self.tstindices)

            for i in range(1, self.numVideo):
                self.trainindices_total = np.hstack((self.trainindices_total, self.trainindices+self.numFrame*i))
                self.valindices_total = np.hstack((self.valindices_total, self.valindices+self.numFrame*i))
                self.tstindices_total = np.hstack((self.tstindices_total, self.tstindices+self.numFrame*i))

            np.random.shuffle(self.trainindices_total)
            np.random.shuffle(self.valindices_total)
            np.random.shuffle(self.tstindices_total)

            # h5data_seqpos = h5data_seqpos[indices]
            #
            if train == 'train':
                self.pc_data_index = self.trainindices_total
                self.pc_valid_index = self.trainindices_total
            elif train == 'test':
                self.pc_data_index = self.tstindices_total
            else:
                self.pc_data_index = self.valindices_total




        self.pc_label_index = None
        return

    def __len__(self):
        return len(self.pc_data_index)

    def __getitem__(self, idx):
        sampleindex_seq = self.pc_data_index[idx]//self.numFrame
        sampleindex_seq = self.validVideoInd[sampleindex_seq]
        sampleindex_frame = self.pc_data_index[idx]%self.numFrame

        if self.pc_label_index is None:
            pcl_data = self.full_data[self.pcl_key][sampleindex_seq, sampleindex_frame, :].reshape(self.numPoint,3).T.astype(self.float_type)
            pcl_data_nextframe = self.full_data[self.pcl_key][sampleindex_seq, sampleindex_frame + 1, :].reshape(self.numPoint,3).T.astype(self.float_type)

            return pcl_data, pcl_data_nextframe
        else:
            pcl_data = self.full_data[self.pcl_key][sampleindex_seq, sampleindex_frame, :].reshape(self.numPoint,3).T.astype(self.float_type)
            pcl_data_nextframe = self.full_data[self.pcl_key][sampleindex_seq, sampleindex_frame + 1, :].reshape(self.numPoint,3).T.astype(self.float_type)

            return pcl_data, pcl_data_nextframe

if __name__ == '__main__':
    dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_dense_3435_bag72_a0_speed1_20200225.h5', mode='unseen2')
    print('Training -- Number of data:{}'.format(len(dataset)))
    valid_dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_dense_3435_bag72_a0_speed1_20200225.h5', train='valid',mode='unseen2')
    print('Validation -- Number of data:{}'.format(len(valid_dataset)))
    test_dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_dense_3435_bag72_a0_speed1_20200225.h5', train='test',mode='unseen2')
    print('Testing -- Number of data:{}'.format(len(test_dataset)))


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

    # dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_dense_3435_bag72_a0_speed1_20200225.h5', mode='unseen2')
    # print('Training -- Number of data:{}'.format(len(dataset)))
    # valid_dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_dense_3435_bag72_a0_speed1_20200225.h5', train='valid',mode='unseen2')
    # print('Validation -- Number of data:{}'.format(len(valid_dataset)))
    # test_dataset = PointCloudDataSetFromH5_3435_seq_interpolation('/home/zehang/Downloads/dataset/bagdata_dense_3435_bag72_a0_speed1_20200225.h5', train='test',mode='unseen2')
    # print('Testing -- Number of data:{}'.format(len(test_dataset)))
    # model.fit(dataset, valid_dataset, batch_size=16, n_epoch=100, lr=3e-4, denoising=0.002, save_every=10, visdom=False)

