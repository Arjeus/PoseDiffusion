import numpy as np 
import open3d as o3d
import glob
# import pdb
glob = glob.glob("/home/arj/code/datasets/pcd_train/*.pcd")
for file in glob:
    pcd = o3d.io.read_point_cloud(file)
    out_arr = np.asarray(pcd.points)  
    # pad the array by adding 3 more columns with zeros
    # pdb.set_trace()
    if out_arr.shape[1] < 6:
        out_arr = np.pad(out_arr, ((0,0), (0, 6-out_arr.shape[1])), 'constant', constant_values=0)
    # pad the array to (60000, 3)
    if out_arr.shape[0] < 60000:
        out_arr = np.pad(out_arr, ((0, 60000-out_arr.shape[0]), (0,0)), 'edge')
    np.save(file.split(".")[0]+".npy", out_arr)