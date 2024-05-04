#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <pybind11/eigen.h>

namespace py = pybind11;

py::array_t<float> transform_to_pointnet(py::array_t<float> input_array) {
    py::buffer_info buf_info = input_array.request();
    int rows = buf_info.shape[0];
    int cols = buf_info.shape[1];

    float *ptr = static_cast<float *>(buf_info.ptr);

    Eigen::MatrixXf cloud = Eigen::MatrixXf::Map(ptr, rows, cols);
    // parameters
    float block_size = 1.0f;
    float stride = 0.5f;
    int block_points = 4096;
    float padding = 0.001f;

    // transform to Eigen
    Eigen::MatrixXf scenepointsxx = cloud.leftCols(3);
    Eigen::MatrixXf ones = Eigen::MatrixXf::Ones(scenepointsxx.rows(), 3);
    Eigen::MatrixXf scenepointsx(scenepointsxx.rows(), scenepointsxx.cols() + ones.cols());
    scenepointsx << scenepointsxx, ones;

    Eigen::RowVectorXf coord_min = scenepointsx.colwise().minCoeff();
    Eigen::RowVectorXf coord_max = scenepointsx.colwise().maxCoeff();
    // Eigen::VectorXf labelweights = Eigen::VectorXf::Ones(13);
    Eigen::MatrixXf point_set_ini = scenepointsx;
    Eigen::MatrixXf points = point_set_ini;
    // coord_min = points.leftCols(3).colwise().minCoeff();
    // coord_max = points.leftCols(3).colwise().maxCoeff();
    int grid_x = static_cast<int>(std::ceil((coord_max(0) - coord_min(0) - block_size) / stride)) + 1;
    int grid_y = static_cast<int>(std::ceil((coord_max(1) - coord_min(1) - block_size) / stride)) + 1;
    printf("grid_x: %d, grid_y: %d\n", grid_x, grid_y);
    Eigen::MatrixXf data_room;
    std::vector<Eigen::VectorXi> index_room;
    std::random_device rd;
    std::mt19937 gen(rd());
    for (int index_y = 0; index_y < grid_y; ++index_y) {
        for (int index_x = 0; index_x < grid_x; ++index_x) {
            float s_x = coord_min(0) + index_x * stride;
            float e_x = std::min(s_x + block_size, coord_max(0));
            s_x = e_x - block_size;
            float s_y = coord_min(1) + index_y * stride;
            float e_y = std::min(s_y + block_size, coord_max(1));
            s_y = e_y - block_size;
        std::vector<int> temp_idxs;

            // Iterate through each point to check if it lies within the padded boundary
            for (int i = 0; i < points.rows(); ++i) {
                if (points(i, 0) >= s_x - padding && points(i, 0) <= e_x + padding &&
                    points(i, 1) >= s_y - padding && points(i, 1) <= e_y + padding) {
                    temp_idxs.push_back(i);
                }
            }

            // Convert std::vector<int> to Eigen::VectorXi
            Eigen::VectorXi point_idxs(temp_idxs.size());
            for (size_t i = 0; i < temp_idxs.size(); ++i) {
                point_idxs(i) = temp_idxs[i];
            }

            if (point_idxs.size() == 0) {
                continue;
            }
            int num_batch = static_cast<int>(std::ceil(point_idxs.size() / static_cast<float>(block_points)));
            int point_size = num_batch * block_points;

            bool replace = (point_size - point_idxs.size() <= point_idxs.size()) ? false : true;
            Eigen::VectorXi point_idxs_repeat(point_size - point_idxs.size());
            if (replace) {
                // With replacement: Randomly pick elements and allow for the same element to be picked more than once
                std::uniform_int_distribution<> dis(0, point_idxs.size() - 1);
                for (int i = 0; i < point_idxs_repeat.size(); ++i) {
                    // std::uniform_int_distribution<> dis(0, point_idxs.size() - 1);
                    point_idxs_repeat(i) = point_idxs(dis(gen));
                }
            } else {
                // Without replacement: Shuffle and pick the first (point_size - point_idxs.size()) elements
                // Make a copy of the original indices to shuffle
                std::vector<int> temp_idxs(point_idxs.data(), point_idxs.data() + point_idxs.size());
                std::shuffle(temp_idxs.begin(), temp_idxs.end(), gen);
                for (int i = 0; i < point_idxs_repeat.size(); ++i) {
                    point_idxs_repeat(i) = temp_idxs[i];
                }
            }
            int total_size = point_idxs.size() + point_idxs_repeat.size();
            Eigen::VectorXi concatenated_points(total_size);

            // Concatenate using a loop (more efficient for Eigen vectors)
            int original_size = point_idxs.size();
            int additional_size = point_idxs_repeat.size();
            point_idxs.conservativeResize(original_size + additional_size);
            point_idxs.segment(original_size, additional_size) = point_idxs_repeat;

            std::shuffle(point_idxs.begin(), point_idxs.end(), gen);
            Eigen::MatrixXf data_batch(point_idxs.size(), points.cols());

            for (int i = 0; i < point_idxs.size(); ++i)
            {
                data_batch.row(i) = points.row(point_idxs(i));
            }
            
            Eigen::MatrixXf normlized_xyz = Eigen::MatrixXf::Zero(point_size, 3);
            
            if (coord_max(0) != 0)  // Ensure divisor is not zero
                normlized_xyz.col(0) = data_batch.col(0) / coord_max(0);
            if (coord_max(1) != 0)
                normlized_xyz.col(1) = data_batch.col(1) / coord_max(1);
            if (coord_max(2) != 0)
                normlized_xyz.col(2) = data_batch.col(2) / coord_max(2);
            data_batch.col(0).array() -= s_x + block_size / 2.0f;
            data_batch.col(1).array() -= s_y + block_size / 2.0f;
            data_batch.col(3).array() /= 255.0f;
            data_batch.col(4).array() /= 255.0f;
            data_batch.col(5).array() /= 255.0f;
            data_batch.conservativeResize(point_size, data_batch.cols() + 3);
            data_batch.rightCols(3) = normlized_xyz;
            if (data_room.size() == 0) {
                // copy data_batch to data_room
                data_room = data_batch;

            } else {
                // concatenate data_batch to data_room
                Eigen::MatrixXf temp(data_room.rows() + data_batch.rows(), data_room.cols());
                temp << data_room, data_batch;
                data_room = temp;
            }
        }
    }

    
    int num_features = 9;

   // Reshaping in C++ doesn't work like in Python; you'd handle it logically
    int num_blocks = data_room.rows();
    
    Eigen::MatrixXf batch_data(1, block_points * num_features); // 1 x (block_points * num_features)
    Eigen::VectorXf batch_point_index(block_points); // block_points x 1
    Eigen::VectorXf batch_smpw(block_points); // block_points x 1

    // Simulate data processing
    int start_idx = 0;
    int end_idx = std::min(1, num_blocks);

    int real_batch_size = end_idx - start_idx;
    for (int i = 0; i < real_batch_size * block_points; ++i) {
        for (int j = 0; j < num_features; ++j) {
            batch_data(0, i * num_features + j) = data_room(start_idx * block_points + i, j);
        }
    }

    // Transpose operation - this transposes the entire block of data
    Eigen::MatrixXf transposed_data = Eigen::Map<Eigen::MatrixXf>(batch_data.data(), num_features, block_points).transpose();
    //return transposed_data as a numpy array
    return py::array_t<float>(transposed_data.size(), transposed_data.data());
}

PYBIND11_MODULE(transform_to_pointnet, m) {
    m.doc() = "converts pointcloud data to pointnet features"; // optional module docstring

    m.def("cloud_to_pointnet", &transform_to_pointnet, "converts pointcloud data to pointnet features");
}