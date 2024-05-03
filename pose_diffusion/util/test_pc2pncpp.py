import numpy as np
import open3d as o3d
import transform_to_pointnet
import pdb
a = o3d.io.read_point_cloud("/home/arj/code/datasets/pcd_train/Bldg1_Stage3_Spot265.450.ply")
pnet = transform_to_pointnet.cloud_to_pointnet(np.asarray(a.points))
# reshape a to (9,4096)
pdb.set_trace()
final = pnet.reshape((9,4096))
print(final.shape)