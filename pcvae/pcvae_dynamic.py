'''
PointCloudVariationalAutoEncoder
'''

from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

import numpy as np

from pointnet import PointNetfeat, feature_transform_reguliarzer
from pcn_decoder import PCNDecoder
from ChamferDistancePyTorch import chamfer_distance_with_batch, centroidPointCloud
# from CentroidPointCloud import centroidPointCloud
from utils import batch_rodriguez_formula_elementwise
from visdom_utils import VisdomInterface

# from pygcem import GaussianCEM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PointCloudVariationalAutoEncoderDisplacement(nn.Module):
    def __init__(self, n_points=2048, latent_size=64, use_pcn=0, mode=False):
        super(PointCloudVariationalAutoEncoderDisplacement, self).__init__()

        self.n_points = n_points
        self.latent_size = latent_size
        self.use_pcn = use_pcn

        self.use_pcn = use_pcn
        self.r1 = None
        self.r2 = None
        self.mode = None
        self.numValidFrame = None
        self.displacementmode = mode


        #prepare encoder net, use PointNet structure.
        self.pointnet_feature = PointNetfeat(global_feat=True, feature_transform=False)

        self.enc_feature = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.encode_transform_mu = nn.Linear(256, latent_size)
        self.encode_transform_logvar = nn.Linear(256, latent_size)

        #prepare decoder net, full fc
        if not use_pcn:
            self.fc_feature = nn.Sequential(
                nn.Linear(latent_size, 256),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(256, 256),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(256, n_points*3)
            )
            self.n_points_coarse = 0
        else:
            grid_size=use_pcn
            self.n_points_coarse = int(n_points/grid_size**2)
            self.pcn_decorder = PCNDecoder(latent_size, self.n_points_coarse, grid_size=grid_size, r=0.01)

        def init_xavier_normal(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
        if not use_pcn:
            self.fc_feature.apply(init_xavier_normal)

        self.encode_transform_mu.apply(init_xavier_normal)
        self.encode_transform_logvar.apply(init_xavier_normal)

        return

    def encode(self, x):
        feat, trans, _ = self.pointnet_feature(x)
        feat = self.enc_feature(feat)
        mu = self.encode_transform_mu(feat)
        logvar = self.encode_transform_logvar(feat)
        return mu, logvar, trans

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, pcn=True):
        if not self.use_pcn:
            x_hat = self.fc_feature(z).view(-1, 3, self.n_points)
            return x_hat
        else:
            x_hat_fine, x_hat_coarse = self.pcn_decorder(z)
            return x_hat_fine, x_hat_coarse

    def fit(self, dataset, valid_dataset=None, batch_size=64, n_epoch=500, lr=5e-4, denoising=0.01, visdom=True, save_every=50, outf='./models'):

        # self.r1 = dataset.lefthandind
        # self.r2 = dataset.righthandind
        self.mode = dataset.mode
        self.numValidFrame = dataset.numValidFrame

        print(self.r1)

        train_loader=torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        if self.mode == 'unseen2':
            # long term estimation test
            if valid_dataset is not None:
                valid_loader=torch.utils.data.DataLoader(
                    dataset=valid_dataset,
                    batch_size=1,
                    shuffle=False
                )
            else:
                valid_loader = None
        else:
            if valid_dataset is not None:
                valid_loader=torch.utils.data.DataLoader(
                    dataset=valid_dataset,
                    batch_size=batch_size,
                    shuffle=False # True
                )
            else:
                valid_loader = None

        opt = optim.Adam(self.parameters(), lr=lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, int(n_epoch/3), gamma=0.5, last_epoch=-1)

        if visdom:
            vis = VisdomInterface(port=8097)

        recons_epoch_loss = []
        valid_epoch_loss = []
        transreg_epoch_loss = []
        kld_epoch_loss = []

        valid_epoch_loss_shift = []

        #############################################
        # recons_epoch_loss_shift = []

        # chamfer_diff_baseline_copy_toal = []
        valid_chamfer_diff_baseline_copy_toal = []

        valid_chamfer_diff_baseline_copy_toal_shift = []



        def train_epoch(epoch, beta):
            recons_batch_loss = []
            transreg_batch_loss = []
            kld_batch_loss = []

            #############################################
            # recons_batch_loss_shift = []


            offset_avg = None

            # chamfer_diff_baseline_copy_batch = []

            # for batch_idx, (data, _) in enumerate(train_loader):
            for batch_idx, data in enumerate(train_loader):
                data_xt = data[0]
                data_xtnext = data[1]

                # don't use cuda here
                if batch_idx == 0:
                    # offset_avg = (data_xtnext - data_xt).sum(axis=0)
                    offset_avg = torch.mean(centroidPointCloud(data_xtnext) - centroidPointCloud(data_xt), axis=0).data
                    # print(offset_avg.mean(axis=0).shape)
                else:
                    # offset_avg = offset_avg + (data_xtnext - data_xt).sum(axis=0)
                    offset_avg += torch.mean(centroidPointCloud(data_xtnext) - centroidPointCloud(data_xt), axis=0).data


                data_xt = Variable(data_xt).to(device)
                samplenoise = torch.randn(data_xt.shape)*denoising
                x = data_xt + samplenoise.to(device)
                # x = Variable(data_xt + torch.randn(data_xt.shape)*denoising).to(device)

                data_xtnext = data_xtnext.to(device)

                x_hat, x_hat_coarse, _, mu, logvar, trans = self.forward(x)
                
                if self.displacementmode is True:
                    x_hat = x_hat + x

                chamfer_loss = self.reconstruction_loss(data_xtnext, x_hat)

                #############################################
                # chamfer_loss_shift = self.recons_batch_loss_shift(data_xtnext, x_hat)


                # print(offset_avg.shape)
                    # print(offset_avg.mean(axis=0).shape)

                # # simple copy
                # chamfer_diff_baseline_copy = self.reconstruction_loss(data_xtnext, data_xt)

                if self.use_pcn:
                    x_hat_coarse = x_hat_coarse + x
                    chamfer_loss_coarse = self.reconstruction_loss(data_xtnext, x_hat_coarse)
                else:
                    chamfer_loss_coarse = 0

                kld_loss = self.kld_loss(mu, logvar)
                trans_reg_loss = feature_transform_reguliarzer(trans)
                loss = self.n_points*chamfer_loss + trans_reg_loss + kld_loss * beta * self.latent_size + chamfer_loss_coarse * self.n_points_coarse

                recons_batch_loss.append(chamfer_loss.item())
                transreg_batch_loss.append(trans_reg_loss.item())
                kld_batch_loss.append(kld_loss.item())

                # # simple copy
                # chamfer_diff_baseline_copy_batch.append(chamfer_diff_baseline_copy.item())

                opt.zero_grad()
                loss.backward()
                opt.step()

                if batch_idx % 25 == 0 and True:        #suppress iteration output now
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Chamfer Loss: {:.6f}; KLD Loss: {:.6f}; Trans Reg Loss: {:.6f};'.format(
                        epoch, batch_idx * len(data[0]), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), chamfer_loss.item(), kld_loss.item(), trans_reg_loss.item()
                        ))

            recons_epoch_loss.append(np.mean(recons_batch_loss))
            transreg_epoch_loss.append(np.mean(transreg_batch_loss))
            kld_epoch_loss.append(np.mean(kld_batch_loss))

            # # simple copy
            # chamfer_diff_baseline_copy_toal.append(np.mean(chamfer_diff_baseline_copy_batch))

            # return
            # offset_avg = (offset_avg/len(dataset))
            offset_avg = (offset_avg/(batch_idx+1))

            return offset_avg

        for i in range(n_epoch):
            beta = i / float(n_epoch) * 0.9
            beta = 0.9
            
            # train mode
            self.pointnet_feature.train()
            self.enc_feature.train()
            self.encode_transform_mu.train()
            self.encode_transform_logvar.train()
            self.fc_feature.train()
            
            offset_avg = train_epoch(i, beta)
            print(offset_avg)
            offset_avg = offset_avg.to(device)
            scheduler.step()
            
            # eval mode
            self.pointnet_feature.eval()
            self.enc_feature.eval()
            self.encode_transform_mu.eval()
            self.encode_transform_logvar.eval()
            self.fc_feature.eval()

            if valid_loader is not None:
                valid_batch_loss = []
                chamfer_baseline_copy_batch = []

                # shift comparision
                valid_batch_loss_shift = []
                chamfer_baseline_copy_batch_shift = []

                # for _, (val_data, _) in enumerate(valid_loader):
                for batch_idx, val_data in enumerate(valid_loader):

                    valdata_xt_cpu = val_data[0]
                    valdata_xtnext_cpu = val_data[1]


                    data_xtnext = valdata_xtnext_cpu.to(device) # target

                    # if self.extrapolation_accumulate is False:
                    #     x = Variable(valdata_xt_cpu).to(device)  # input
                    # else:
                    #     x = x_hat
                    if self.mode != 'unseen2':
                        x = Variable(valdata_xt_cpu).to(device)
                    else:
                        if ((batch_idx % self.numValidFrame)%5) == 0:
                            x = Variable(valdata_xt_cpu).to(device)  # input
                        else:
                            # need to be exacltly the same lengh of differnt batches... so batch = 1
                            x = x_hat

                    # x = Variable(valdata_xt_cpu).to(device) # input
                    # regress the residual/offset
                    x_hat, _, _, _, _, _ = self.forward(x)
                    if self.displacementmode is True:
                        x_hat = x_hat + x
                    x_hat = x_hat.detach()

                    chamfer_loss = self.reconstruction_loss(data_xtnext, x_hat)

                    # simple copy
                    # CD with shifting, evaluate the internal deformation, r1 r2 none; r1 r2 only for particle, not for point cloud
                    chamfer_loss_shift = self.reconstruction_loss_shift(data_xtnext, x_hat, self.r1, self.r2)

                    # valid_batch_loss.append(chamfer_loss.item())
                    valid_batch_loss.append(chamfer_loss.item())
                    ################ shift
                    valid_batch_loss_shift.append(chamfer_loss_shift.item())

                    # copy
                    if self.mode != 'unseen2':
                        chamfer_baseline_copy = self.reconstruction_loss(data_xtnext.data, x+offset_avg.data)
                        chamfer_baseline_copy_batch.append(chamfer_baseline_copy.item())
                        chamfer_baseline_copy_shift = self.reconstruction_loss_shift(data_xtnext.data, x + (offset_avg).data, self.r1, self.r2)
                        chamfer_baseline_copy_batch_shift.append(chamfer_baseline_copy_shift.item())
                    else:
#                         if (batch_idx % self.numValidFrame) == 0:
                        if ((batch_idx % self.numValidFrame)%5) == 0:
                            chamfer_baseline_copy = self.reconstruction_loss(data_xtnext.data, x+offset_avg.data)
                            chamfer_baseline_copy_batch.append(chamfer_baseline_copy.item())
                            chamfer_baseline_copy_shift = self.reconstruction_loss_shift(data_xtnext.data, x + (offset_avg).data, self.r1, self.r2)
                            chamfer_baseline_copy_batch_shift.append(chamfer_baseline_copy_shift.item())
                            x_pre_copy = x.data + offset_avg.data
                        else:
                            chamfer_baseline_copy = self.reconstruction_loss(data_xtnext.data, x_pre_copy+offset_avg.data)
                            chamfer_baseline_copy_batch.append(chamfer_baseline_copy.item())
                            chamfer_baseline_copy_shift = self.reconstruction_loss_shift(data_xtnext.data, x_pre_copy + (offset_avg).data, self.r1, self.r2)
                            chamfer_baseline_copy_batch_shift.append(chamfer_baseline_copy_shift.item())
                            x_pre_copy = x_pre_copy + offset_avg.data


                valid_epoch_loss.append(np.mean(valid_batch_loss))
                valid_chamfer_diff_baseline_copy_toal.append(np.mean(chamfer_baseline_copy_batch))

                ####################################################3
                valid_epoch_loss_shift.append(np.mean(valid_batch_loss_shift))
                valid_chamfer_diff_baseline_copy_toal_shift.append(np.mean(chamfer_baseline_copy_batch_shift))


                print('Validation Error: Chamfer Loss - {:.6f}; Simple copy CD: {:.6f}'.format(valid_epoch_loss[-1], valid_chamfer_diff_baseline_copy_toal[-1]))
                print('Validation Error with shift: shifting CD - {:.6f}; Simple copy shifting CD: {:.6f}'.format(valid_epoch_loss_shift[-1], valid_chamfer_diff_baseline_copy_toal_shift[-1]))

            if visdom:
                vis.update_losses(recons_epoch_loss, transreg_epoch_loss, kld_epoch_loss, valid_epoch_loss,
                                  train_recon_copy=None, val_recon_copy=valid_chamfer_diff_baseline_copy_toal)

            if i == 0 or (i+1) % save_every == 0:
                torch.save(self.state_dict(), '%s/pcvae_epoch_%d.pth' % (outf, i))
        return

    def forward(self, x):
        mu, logvar, trans = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if self.use_pcn:
            x_hat, x_hat_coarse = self.decode(z)
        else:
            x_hat = self.decode(z)
            x_hat_coarse = None
        return x_hat, x_hat_coarse, z, mu, logvar, trans

    def reconstruction_loss(self, x, x_hat):
        chamfer_loss_1, chamfer_loss_2 = chamfer_distance_with_batch(x, x_hat)
        return chamfer_loss_1.mean() + chamfer_loss_2.mean()

    def reconstruction_loss_shift(self, x, x_hat, r1=None, r2=None):
        # only r1 and r2 in particle dataset, not point cloud resample dataset
        if (r1 is None) or (r2 is None):
            dist_centroid = centroidPointCloud(x) - centroidPointCloud(x_hat)
        else:
            dist_centroid_r1 = centroidPointCloud(x[:,:,r1]) - centroidPointCloud(x_hat[:,:,r1])
            # dist_centroid_r2 = centroidPointCloud(x[:,:,r2]) - centroidPointCloud(x_hat[:,:,r2])
            dist_centroid = dist_centroid_r1
            # print(x[:,:,r1].shape)


        chamfer_loss_1, chamfer_loss_2 = chamfer_distance_with_batch(x, x_hat+dist_centroid)
        return chamfer_loss_1.mean() + chamfer_loss_2.mean()

    def kld_loss(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def load_model(self, mdl_fname, cuda=False):
        if cuda:
            self.load_state_dict(torch.load(mdl_fname))
            self.cuda()
        else:
            self.load_state_dict(torch.load(mdl_fname, map_location='cpu'))
        self.eval()

    def fit_shapes(self, input_shapes, init_from_input=True, n_iters=10, lr=.1, ll_weight=1.0, verbose=False):
        if init_from_input:
            encode_z = self.encode(input_shapes)[0]
        else:
            encode_z = z = torch.randn(input_shapes.shape[0], self.latent_size)
            if next(self.parameters()).is_cuda:
                encode_z = encode_z.cuda()

        encode_z = Variable(encode_z, requires_grad=True)

        # #grad,  = torch.autograd.grad(tol_loss, encode_z, create_graph=True)

        #try adam optimizer
        solver = optim.Adam([encode_z], lr=lr, weight_decay=0)
        for i in range(n_iters):
            #print(encode_z)
            decode_x = self.decode(encode_z)[0]
            loss_input_to_decode, loss_decode_to_input = chamfer_distance_with_batch(input_shapes, decode_x)     #this allows small cost when decoded_x contains input_shapes
            prior_nll = encode_z.norm(dim=1)
            tol_loss = (loss_input_to_decode.sum() + loss_decode_to_input.sum()*0.05) + ll_weight * prior_nll.sum()
            if verbose:
                print('Iteration {} - Reconstruction Loss/Log-Likelihood: {}/{}'.format(i+1, loss_input_to_decode.sum().item(), -prior_nll.sum().item()))
            solver.zero_grad()
            tol_loss.backward()
            solver.step()

        return self.decode(encode_z)[0]

    def fit_alignment(self, input_shapes, lr=.01, n_iters=20, verbose=False):
        '''
        try to align the input_shapes with latent variables: register before/after?
        '''
        trans = torch.zeros(input_shapes.shape[0], 3, dtype=input_shapes.dtype)
        rot = torch.tensor(np.array([[1, 0, 0, 0] for _ in range(input_shapes.shape[0])]), dtype=input_shapes.dtype)     #axis-angle representation [axis, angle]

        if input_shapes.is_cuda:
            trans = trans.cuda()
            rot = rot.cuda()

        trans = Variable(trans, requires_grad=True)
        rot = Variable(rot, requires_grad=True)

        #try to solve the trans and rot
        solver = optim.Adam([trans, rot], lr=lr, weight_decay=0)
        for i in range(n_iters):
            aligned_shapes = batch_rodriguez_formula_elementwise(input_shapes, rot) + trans[:, :, None]
            encode_z = self.encode(aligned_shapes)[0]
            decoded_shapes = self.decode(encode_z)[0]
            loss_input_to_decode, loss_decode_to_input = chamfer_distance_with_batch(aligned_shapes, decoded_shapes)
            loss = loss_input_to_decode.sum() + .01 * loss_decode_to_input.sum()

            if verbose:
                print('Iteration {} - Alignment Loss: {}'.format(i+1, loss.item()))

            solver.zero_grad()
            loss.backward()
            solver.step()

        return batch_rodriguez_formula_elementwise(input_shapes, rot) + trans[:, :, None], trans, rot

    def sample(self, n_sample=1, z=None, detach=True):
        if z is None:
            z = torch.randn(n_sample, self.latent_size)
        if next(self.parameters()).is_cuda:
            z = z.cuda()
        if self.use_pcn > 0:
            return self.decode(z)[0].detach() if detach else self.decode(z)[0]
        else:
            return self.decode(z).detach() if detach else self.decode(z)
