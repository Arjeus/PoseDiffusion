import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
sys.path.append('/home/arjay55/code/Pointnet_Pointnet2_pytorch/')

import os
import numpy as np
import glob
import tqdm
import torch.utils.data as torch_data
import provider

from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

def get_unlabeled_data(folder):
    unlabeled_data = []
    for file in glob.glob(folder + "/*.npy"):
        unlabeled_data.append(np.load(file))
    return unlabeled_data


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 9 + 3, [32, 32, 64], False)
        # self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 6 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points)), inplace=True))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

if __name__ == '__main__':

    block_size = 1.0
    stride = 0.5
    block_points = 4096
    padding = 0.001

    model = get_model(13)
    checkpoint = torch.load("/home/arjay55/code/Pointnet_Pointnet2_pytorch/log/sem_seg/pointnet2_sem_seg/checkpoints/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    classifier = model.cuda()
    # how to get to xyz?
    numpy_sample_input = get_unlabeled_data("/home/arjay55/code/pcl_exp_migrate/pcl_exp")
    scene_points = numpy_sample_input[:]
    # fill scene_points with added 3 channels
    scene_points[0] = np.concatenate((scene_points[0], np.ones((scene_points[0].shape[0], 3))), axis=1)
    coord_min, coord_max = np.amin(scene_points, axis=0)[:], np.amax(scene_points, axis=0)[:]
    labelweights = np.ones(13)
    point_set_ini = scene_points
    points = point_set_ini[0]
    coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
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
    batch_point_index[0:real_batch_size, ...] = index_room[start_idx:end_idx, ...]
    # batch_smpw[0:real_batch_size, ...] = sample_weight[start_idx:end_idx, ...]          
    torch_data = torch.Tensor(batch_data)
    torch_data = torch_data.float().cuda()
    torch_data = torch_data.transpose(2, 1)
    seg_pred, _ = classifier(torch_data)
    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()
    print(batch_pred_label)
    print(batch_pred_label.shape)
    print(batch_point_index)
    print(batch_point_index.shape)

