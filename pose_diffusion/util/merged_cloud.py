import pdb
import open3d as o3d
import json
import numpy as np
import os

def transform_to_pointnet(cloud):
    # parameters
    block_size = 1.0
    stride = 0.5
    block_points = 4096
    padding = 0.001

    # transform to numpy
    data = np.asarray(cloud.points)
    scenepointsx = data[:, :3]
    scenepointsx = np.concatenate((scenepointsx, np.ones((scenepointsx.shape[0], 3))), axis=1)
    coord_min, coord_max = np.amin(scenepointsx, axis=0)[:], np.amax(scenepointsx, axis=0)[:]
    labelweights = np.ones(13)
    point_set_ini = scenepointsx
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
    
    return batch_data

def add_merged_clouds(batch):
    # execute each row as it is a batch.
    # transpose the batch["cloud_path"]
    rowlength = len(batch["cloud_path"][0])
    new_cloud_path = []
    for _ in range(rowlength):
        new_cloud_path.append([])

    for i in range(len(batch["cloud_path"])):
        for j in range(rowlength):
            new_cloud_path[j].append(batch["cloud_path"][i][j])


    merged_cloud_net_list = []
    for i in range(len(new_cloud_path)): # len(batch is different)
        merged_cloud = o3d.geometry.PointCloud()
        for idx,path in enumerate(new_cloud_path[i]):
            cloud = o3d.io.read_point_cloud(path)
            #apply transformations using 
            cloud = cloud.translate(batch["T"][i][idx].numpy())
            cloud = cloud.rotate(batch["R"][i][idx].numpy())
            merged_cloud += cloud

        # downsample pointcloud to 0.05
        merged_cloud = merged_cloud.voxel_down_sample(voxel_size=0.05)
        merged_cloud_net = transform_to_pointnet(merged_cloud)
    
    #concatenate merged_cloud_net to batch["image"]
    merged_cloud_net = merged_cloud_net.unsqueeze(1)
    batch["image"] = torch.cat((batch["image"], merged_cloud_net), 1)
    
 