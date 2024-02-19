import numpy as np
import open3d as o3d
import glob
import os  # Import for path handling
import pdb

pcd_files = glob.glob("/home/arj/code/datasets/pcd_train/*.pcd")
# delete all npy files
for file in glob.glob("/home/arj/code/datasets/pcd_train/*.npy"):
    os.remove(file)

def get_unlabeled_data(folder):
    unlabeled_data = []
    for file in glob.glob(os.path.join(folder, "*.npy")):
        data = np.load(file)
        xyz = data[:, :3]
        unlabeled_data.append(xyz)
    return unlabeled_data


for file in pcd_files:
    pcd = o3d.io.read_point_cloud(file)
    out_arr = np.asarray(pcd.points)  
    np.save(file.split(".")[0]+".npy", out_arr)
    
block_size = 1.0
stride = 0.5
block_points = 4096
padding = 0.001

# how to get to xyz?
numpy_sample_input = get_unlabeled_data("/home/arj/code/datasets/pcd_train")
scene_points = numpy_sample_input[:]
# fill scene_points with added 3 channels
for id in range(len(scene_points)):
    scene_points[id] = np.concatenate((scene_points[id], np.ones((scene_points[id].shape[0], 3))), axis=1)
    coord_min, coord_max = np.amin(scene_points[id], axis=0)[:], np.amax(scene_points[id], axis=0)[:]
    labelweights = np.ones(13)
    point_set_ini = scene_points[id]
    points = point_set_ini
    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3] # id here?
    grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - block_size) / stride) + 1)
    grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - block_size) / stride) + 1)   
    data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]),  np.array([])
    for index_y in range(0, grid_y):
        for index_x in range(0, grid_x):
            s_x = coord_min[0] + index_x * stride
            e_x = min(s_x + block_size, coord_max[0])
            s_x = e_x - block_size
            s_y = coord_min[1] + index_y * stride
            e_y = min(s_y + block_size, coord_max[1])
            s_y = e_y - block_size        
            point_idxs = np.where(
                (points[:, 0] >= s_x - padding) & (points[:, 0] <= e_x + padding) & (points[:, 1] >= s_y - padding) & (
                            points[:, 1] <= e_y + padding))[0]
            if point_idxs.size == 0:
                continue
            num_batch = int(np.ceil(point_idxs.size / block_points))
            point_size = int(num_batch * block_points)
            replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
            point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
            point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
            np.random.shuffle(point_idxs)
            data_batch = points[point_idxs, :]
            normlized_xyz = np.zeros((point_size, 3))
            normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
            normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
            normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
            data_batch[:, 0] = data_batch[:, 0] - (s_x + block_size / 2.0)
            data_batch[:, 1] = data_batch[:, 1] - (s_y + block_size / 2.0)
            data_batch[:, 3:6] /= 255.0
            data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
            data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
            index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs

    data_room = data_room.reshape((-1, block_points, data_room.shape[1]))
    # label_room = label_room.reshape((-1, block_points))
    # sample_weight = sample_weight.reshape((-1, block_points))
    index_room = index_room.reshape((-1, block_points))

    num_blocks = data_room.shape[0]
    batch_data = np.zeros((1, block_points, 9))
    batch_point_index = np.zeros((1, block_points))
    batch_smpw = np.zeros((1, block_points))
    start_idx = 0*1
    end_idx = min((0 + 1) * 1, num_blocks)
    real_batch_size = end_idx - start_idx 
    batch_data[0:real_batch_size, ...] = data_room[start_idx:end_idx, ...]
    # batch_smpw[0:real_batch_size, ...] = sample_weight[start_idx:end_idx, ...]   
    # remove first dimension of batch_data
    batch_data = batch_data[0]
    batch_data = batch_data.transpose()
    # save to npy
    filename = "/home/arj/code/datasets/pcd_train/pointnet_ready_{}.npy".format(id)
    np.save(filename, batch_data)
    