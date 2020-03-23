import numpy as np
import open3d as o3d
# import trimesh
import h5py
import time
import pcvae
import utils
import numpy as np

import torch
from pcvae_dynamic import PointCloudVariationalAutoEncoderDisplacement


# load the bagdata.h5
# full_data = h5py.File('/home/zehang/Downloads/dataset/bagdata_dense_3435_bag72_a0_speed1_20200225.h5', "r")
full_data = h5py.File('/media/lici/depot/dataset_zhweng/bagdata_dense_3435_bag72_a0_speed1_20200225.h5', "r")

numVideo = full_data['posSeqDense_3435'].shape[0]
numFrame = full_data['posSeqDense_3435'].shape[1]
numPoint = full_data['posSeqDense_3435'].shape[2]
numTotalFrame = numVideo*numFrame
# convert to ndarray
h5data_seqpos = full_data['posSeqDense_3435'][0:10].astype(np.float)
print(h5data_seqpos.shape)


def copyBaseline(seq, refresh = 1,  movement_offset = None):
    offset = 3
    copyResult = np.zeros(seq.shape)
    
    if refresh == -1:
        for i in range(seq.shape[0]):
            if i == 0:
                basepos = seq[i, :, :].copy()
                copyResult[i, :, :] = basepos.copy()
            else:
                copyResult[i, :, :] = seq[i-1, :, :] + movement_offset
        
    else:
        for i in range(seq.shape[0]):
            if i%refresh == 0:
                basepos = seq[i, :, :].copy()
                copyResult[i, :, :] = basepos.copy()
            else:
                basepos += movement_offset
                copyResult[i, :, :] = basepos.copy()
        
    # offset for plot
    for i in range(seq.shape[0]):
        basepos = copyResult[i,:,:].copy()
        baseposT = basepos.T
        baseposT[:,1] = baseposT[:,1] - offset
        base = baseposT.T
        copyResult[i,:,:] = base

    return copyResult

def pcvae_recon(seq, refresh = 1):
    offset = 6
    model = PointCloudVariationalAutoEncoderDisplacement(n_points=3435, latent_size=32, use_pcn=0)
    model.load_model('models/unseen2_5_20200303_nondisplacement/pcvae_epoch_99.pth')
    # model.load_model('models/unseen2_5_20200303_full_preaction0/pcvae_epoch_99.pth')
    seqlen, dim, numpoint = seq.shape
    pcvae_recon_Result = np.zeros(seq.shape)
    
    if refresh == -1:
        for i in range(seq.shape[0]):
            if i == 0:
                basepos = seq[i, :, :].copy()
                pcvae_recon_Result[i, :, :] = basepos.copy()
            else:
                nextpredict, _, _, _, _, _ = model.forward(torch.tensor(seq[i-1, :, :].reshape(1, dim, numpoint)).float())
                pcvae_recon_Result[i, :, :] = nextpredict.detach().numpy().copy()
    else:       
        for i in range(seq.shape[0]):
            if i % refresh == 0:
                basepos = seq[i, :, :].copy()
                pcvae_recon_Result[i,:,:] = basepos.copy()
            else:
                nextpredict, _, _, _, _, _ = model.forward(torch.tensor(basepos.reshape(1, dim, numpoint)).float())
                basepos = nextpredict.detach().numpy().copy()
                pcvae_recon_Result[i,:,:] = basepos.copy()
                
    for i in range(seq.shape[0]):
        basepos = pcvae_recon_Result[i,:,:].copy()
        baseposT = basepos.T
        baseposT[:,1] = baseposT[:,1] - offset
        base = baseposT.T
        pcvae_recon_Result[i,:,:] = base
    return pcvae_recon_Result

def pcvae_recon_displacement(seq, refresh = 1):
    offset = 9
    model = PointCloudVariationalAutoEncoderDisplacement(n_points=3435, latent_size=32, use_pcn=0)
    model.load_model('models/unseen2_5_20200301/pcvae_epoch_9.pth')
    # model.load_model('models/unseen2_5_20200303_displacement_preaction0/pcvae_epoch_99.pth')
    seqlen, dim, numpoint = seq.shape
    pcvae_recon_disp_Result = np.zeros(seq.shape)
    
    if refresh == -1:
        for i in range(seq.shape[0]):
            if i == 0:
                basepos = seq[i, :, :].copy()
                pcvae_recon_disp_Result[i, :, :] = basepos.copy()
            else:
                nextpredict, _, _, _, _, _ = model.forward(torch.tensor(seq[i-1, :, :].reshape(1, dim, numpoint)).float())
                pcvae_recon_disp_Result[i, :, :] = nextpredict.detach().numpy() + seq[i-1, :, :]
    else:       
        for i in range(seq.shape[0]):
            if i % refresh == 0:
                basepos = seq[i, :, :].copy()
                pcvae_recon_disp_Result[i,:,:] = basepos.copy()
            else:
                nextpredict, _, _, _, _, _ = model.forward(torch.tensor(basepos.reshape(1, dim, numpoint)).float())
                basepos = nextpredict.detach().numpy() + basepos
                pcvae_recon_disp_Result[i,:,:] = basepos.copy()
                
    for i in range(seq.shape[0]):
        basepos = pcvae_recon_disp_Result[i,:,:].copy()
        baseposT = basepos.T
        baseposT[:,1] = baseposT[:,1] - offset
        base = baseposT.T
        pcvae_recon_disp_Result[i,:,:] = base
    return pcvae_recon_disp_Result


while(1):
    refresh = 1000
    clothid = input("input experiment id (or 'quit') : ")
    if clothid == "quit":
        break
    else:
        '''
        while(1):
            # clothid = input("input experiment id (or 'quit') : ")
            if full_data['initactionid'][int(clothid)] == 0:
                break
            else:
                clothid = input("input experiment id : ")
        '''

    #else:
        repeatTime = int(input("input the repeat time : "))
        speed = float(input("render speed : "))
        # plot the point cloud
        visid = int(clothid)
        # remember to remove the 1st and last frame
        seq = h5data_seqpos[visid,1:-1,:,:].transpose(0,2,1)
        movement_offset = np.array([[-2.3277e-04], [2.0460e-05], [2.0168e-01]])
        seqcopy = copyBaseline(seq, refresh,movement_offset=movement_offset)

        seqpcvae = pcvae_recon(seq,refresh)
        seqpcvae_displacement = pcvae_recon_displacement(seq,refresh)


        
        seq_combine = np.zeros((seq.shape[0], 3, 3435 * 4))
        
        seq_combine[:, :, :3435] = seq
        seq_combine[:, :, 3435:3435*2] = seqcopy
        seq_combine[:, :, 3435*2:3435*3] = seqpcvae
        seq_combine[:, :, 3435*3:] = seqpcvae_displacement
        # seq_combine = np.zeros((38, 3, 3435))     
        # seq_combine[:, :, :] = seqpcvae
        
        #
        # # print(seq.shape)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        print(seq_combine[0].shape)
        totalpoints = seq_combine[0].T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(totalpoints)
        colorpoint = np.zeros(totalpoints.shape)
        pcd.colors = o3d.utility.Vector3dVector(colorpoint)
        vis.add_geometry(pcd)

        ctr = vis.get_view_control()
        ctr.rotate(360, 0.0)

        vis.run()

        # filename = './test'
        # param = ctr.convert_to_pinhole_camera_parameters()
        # trajectory = o3d.camera.PinholeCameraTrajectory()
        # print(param)
        # trajectory.intrinsic = param[0]
        # trajectory.extrinsic = o3d.utility.Matrix4dVector([param[1]])
        # o3d.io.write_pinhole_camera_trajectory(filename, trajectory)
        # vis.destroy_window()
        
        for i in range(repeatTime):
	        for j in range(seq_combine.shape[0]):
	            totalpoints = seq_combine[j].T
	            pcd.points = o3d.utility.Vector3dVector(totalpoints)
	            vis.update_geometry()
	            #vis.update_geometry(pcd)
	            vis.poll_events()
	            vis.update_renderer()
	            vis.run()
	            time.sleep(speed)
        
        vis.destroy_window()
