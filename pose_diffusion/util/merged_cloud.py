import pdb
import open3d as o3d
import json
import numpy as np
import os
import torch
import time
from numba import jit, njit, prange

def transform_to_pointnet_py(cloud):
    # parameters
    block_size = 1.0
    stride = 0.5
    block_points = 8192
    padding = 0.001
    constant_for_normalization = 20

    # transform to numpy
    data = cloud
    scenepointsx = data[:, :3]
    scenepointsx = np.concatenate((scenepointsx, np.ones((scenepointsx.shape[0], 3))), axis=1)
    coord_min, coord_max = np.amin(scenepointsx, axis=0)[:], np.amax(scenepointsx, axis=0)[:]
    labelweights = np.ones(13)
    point_set_ini = scenepointsx[:, :3] 
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
            normlized_xyz[:, 0] = data_batch[:, 0] / constant_for_normalization
            normlized_xyz[:, 1] = data_batch[:, 1] / constant_for_normalization
            normlized_xyz[:, 2] = data_batch[:, 2] / constant_for_normalization
            data_batch[:, 0] = data_batch[:, 0] - (s_x + block_size / 2.0)
            data_batch[:, 1] = data_batch[:, 1] - (s_y + block_size / 2.0)
            data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
            data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
            index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs

    data_room = data_room.reshape((-1, block_points, data_room.shape[1]))
    # label_room = label_room.reshape((-1, block_points))
    # sample_weight = sample_weight.reshape((-1, block_points))
    index_room = index_room.reshape((-1, block_points))

    num_blocks = data_room.shape[0]
    batch_data = np.zeros((1, block_points, 6))  # 6 = 3 original XYZ + 3 normalized XYZ
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

@njit
def process_blocks(points, coord_min, coord_max, grid_x, grid_y, block_size, stride, block_points, padding):
    data_room = []
    index_room = []
    for index_y in prange(grid_y):
        for index_x in prange(grid_x):
            s_x = coord_min[0] + index_x * stride
            e_x = min(s_x + block_size, coord_max[0])
            s_x = e_x - block_size
            s_y = coord_min[1] + index_y * stride
            e_y = min(s_y + block_size, coord_max[1])
            s_y = e_y - block_size        
            point_idxs = np.where(
                (points[:, 0] >= s_x - padding) & 
                (points[:, 0] <= e_x + padding) & 
                (points[:, 1] >= s_y - padding) & 
                (points[:, 1] <= e_y + padding))[0]
            if point_idxs.size == 0:
                continue
            num_batch = int(np.ceil(point_idxs.size / block_points))
            point_size = int(num_batch * block_points)
            replace = point_size - point_idxs.size > point_idxs.size
            point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
            point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
            np.random.shuffle(point_idxs)
            data_batch = points[point_idxs, :]
            normalized_xyz = data_batch[:, :3] / coord_max[:3]
            data_batch[:, :3] -= np.array([s_x + block_size / 2.0, s_y + block_size / 2.0, 0])
            data_batch[:, 3:6] /= 255.0
            data_batch = np.concatenate((data_batch, normalized_xyz), axis=1)
            data_room.append(data_batch)
            index_room.extend(point_idxs)

    return data_room, index_room

@jit(forceobj=True)
def transform_to_pointnet_numba(cloud):
    # parameters
    block_size = 10
    stride = 10
    block_points = 4096
    padding = 0.001

    data = cloud
    scenepointsx = np.concatenate((data[:, :3], np.ones((data.shape[0], 3))), axis=1)
    coord_min, coord_max = np.amin(scenepointsx, axis=0)[:3], np.amax(scenepointsx, axis=0)[:3]
    grid_x = int(np.ceil((coord_max[0] - coord_min[0] - block_size) / stride) + 1)
    grid_y = int(np.ceil((coord_max[1] - coord_min[1] - block_size) / stride) + 1)

    data_room, index_room = process_blocks(scenepointsx, coord_min, coord_max, grid_x, grid_y, block_size, stride, block_points, padding)
    data_room = np.vstack(data_room)
    index_room = np.array(index_room)
    data_room = data_room.reshape((-1, block_points, data_room.shape[1]))
    index_room = index_room.reshape((-1, block_points))
    # num_blocks = data_room.shape[0]
    # batch_data = data_room[:1, ...]  # Simplifying the example to take the first batch
    # batch_data = batch_data.transpose()
    num_blocks = data_room.shape[0]
    batch_data = np.zeros((1, block_points, 9))
    start_idx = 0*1
    end_idx = min((0 + 1) * 1, num_blocks)
    real_batch_size = end_idx - start_idx 
    batch_data[0:real_batch_size, ...] = data_room[start_idx:end_idx, ...]
    batch_data = batch_data[0]
    batch_data = batch_data.transpose()  
    return batch_data

def add_merged_clouds(batch):
    # execute each row as it is a batch.
    # transpose the batch["cloud_path"]
    rowlength = len(batch["cloud_path"][0])
    cloud_list = []
    translation_list = []
    rotation_list = []
    new_cloud_path = []
    fl_list = []
    for _ in range(rowlength):
        new_cloud_path.append([])

    for i in range(len(batch["cloud_path"])):
        for j in range(rowlength):
            new_cloud_path[j].append(batch["cloud_path"][i][j])

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
        # normalize pointcloud
        # merged_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(0.3)) # incidentally, normal estimation distorts pointnet's learning!
        # measure time
        # start_time = time.time()
        # merged_cloud_net = transform_to_pointnet.cloud_to_pointnet(np.asarray(merged_cloud.points))
        merged_cloud_net = transform_to_pointnet_numba(np.asarray(merged_cloud.points))
        # end_time = time.time()
        # print("Time taken for cloud_to_pointnet: ", end_time - start_time)
        merged_cloud_net = torch.from_numpy(merged_cloud_net).unsqueeze(0)
        cloud_list.append((torch.cat((batch["image"][i], merged_cloud_net))).unsqueeze(0))
        # concatenate batch["T"]
        translation_list.append(torch.cat((batch["T"][i], torch.zeros(1,3))).unsqueeze(0))
        # concatenate batch["R"]
        rotation_list.append(torch.cat((batch["R"][i], torch.eye(3).unsqueeze(0))).unsqueeze(0))
        fl_list.append(torch.cat((batch["fl"][i], torch.tensor([3.,3.]).unsqueeze(0))).unsqueeze(0))

        del merged_cloud

    batch["image"] = torch.vstack(cloud_list)
    batch["T"] = torch.vstack(translation_list)
    batch["R"] = torch.vstack(rotation_list)
    batch["fl"] = torch.vstack(fl_list)

    return batch
