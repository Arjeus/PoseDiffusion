import open3d as o3d
import numpy as np
# Load the point cloud
pcd = o3d.io.read_point_cloud("/home/arjay55/code/datasets/site1_handheld3_pcd_raws/site1_handheld3_187.pcd")
#get the bounding box
first_level_voxel_granularity = 1.275
second_level_voxel_granularity = 0.005
max_bound = pcd.get_max_bound()
min_bound = pcd.get_min_bound()

min_x = np.floor(min_bound[0]/first_level_voxel_granularity)
min_y = np.floor(min_bound[1]/first_level_voxel_granularity)
min_z = np.floor(min_bound[2]/first_level_voxel_granularity)

max_x = np.ceil(max_bound[0]/first_level_voxel_granularity)
max_y = np.ceil(max_bound[1]/first_level_voxel_granularity)
max_z = np.ceil(max_bound[2]/first_level_voxel_granularity)


new_data_structure = []
for i in range(int(min_x), int(max_x)):
    for j in range(int(min_y), int(max_y)):
        for k in range(int(min_z), int(max_z)):
            # crop the point cloud
            crop = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=[i*first_level_voxel_granularity, j*first_level_voxel_granularity, k*first_level_voxel_granularity], max_bound=[(i+1)*first_level_voxel_granularity, (j+1)*first_level_voxel_granularity, (k+1)*first_level_voxel_granularity]))
            #voxelize
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(crop,voxel_size=second_level_voxel_granularity)
            if not voxel_grid.is_empty():
                for _ in voxel_grid.get_voxels():
                    second_level_coords = _.grid_index.tolist()
                    new_data_structure.append([i, j, k, *second_level_coords])

array_data_structure = np.array(new_data_structure)

# Save array_data_structure to readable file
np.savetxt('array_data_structure.txt', array_data_structure, fmt='%d')
