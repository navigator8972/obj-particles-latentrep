import numpy as np
import open3d as o3d
# import trimesh
import h5py
import time

# load the bagdata.h5
full_data = h5py.File('../data/bagdata_dense.h5', "r")
numVideo = full_data['posSeqDense_5'].shape[0]
numFrame = full_data['posSeqDense_5'].shape[1]
numPoint = full_data['posSeqDense_5'].shape[2]
numTotalFrame = numVideo*numFrame
# convert to ndarray
h5data_seqpos = full_data['posSeqDense_5'][0:10].astype(np.float)
print(h5data_seqpos.shape)


while(1):
    clothid = input("input experiment id (or 'quit') : ")
    if clothid == "quit":
        break
    else:
        repeatTime = int(input("input the repeat time : "))
        speed = float(input("render speed : "))
        # plot the point cloud
        visid = repeatTime
        seq = h5data_seqpos[visid,:].transpose(0,2,1)
        print(seq.shape)

        # function to read the topo
        def readTopo(meshtopofile):
            return np.array([int(i) for i in open(meshtopofile,'r').read().split(';')[:-1]]).reshape(-1,3)

        # print(seq.shape)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        totalpoints = seq[0].T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(totalpoints)
        colorpoint = np.zeros(totalpoints.shape)
        pcd.colors = o3d.utility.Vector3dVector(colorpoint)
        vis.add_geometry(pcd)
        vis.run()
        
        for i in range(repeatTime):
	        for j in range(seq.shape[0]):
	            totalpoints = seq[j].T
	            pcd.points = o3d.utility.Vector3dVector(totalpoints)
                # vis.update_geometry()
	            vis.update_geometry(pcd)
	            vis.poll_events()
	            vis.update_renderer()
	            vis.run()
	            time.sleep(speed)
                
        vis.destroy_window()
