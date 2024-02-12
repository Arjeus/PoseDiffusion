import numpy as np 
import open3d as o3d
import glob
glob = glob.glob("/home/arj/code/datasets/pcd_train/*.pcd")
for file in glob:
    pcd = o3d.io.read_point_cloud(file)
    out_arr = np.asarray(pcd.points)  
    #save out_arr to npy file
    np.save(file.split(".")[0]+".npy", out_arr)