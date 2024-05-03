import numpy as np
import open3d as o3d
import transform_to_pointnet

a = o3d.io.read_point_cloud("/home/arj/code/datasets/pcd_train/Bldg1_Stage3_Spot265.450.ply")
transform_to_pointnet.cloud_to_pointnet(np.asarray(a.points))
