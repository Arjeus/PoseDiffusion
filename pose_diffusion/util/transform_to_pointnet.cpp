#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

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
    //print grid_y
    std::vector<Eigen::MatrixXf> data_room;
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
            Eigen::Array<bool, Eigen::Dynamic, 1> mask = (points.col(0).array() >= s_x - padding) && (points.col(0).array() <= e_x + padding) && (points.col(1).array() >= s_y - padding) && (points.col(1).array() <= e_y + padding); //verify this
            Eigen::VectorXi point_idxs = Eigen::VectorXi::Zero(mask.count()); 
            int idx = 0;
            for (int i = 0; i < mask.size(); ++i) {
                if (mask(i)) {
                    point_idxs(idx++) = i;
                }
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
                for (int i = 0; i < point_idxs_repeat.size(); ++i) {
                    std::uniform_int_distribution<> dis(0, point_idxs.size() - 1);
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
            int i = 0;
            for (int j = 0; j < point_idxs.size(); ++j) {
                concatenated_points(i++) = point_idxs(j);
            }
            for (int j = 0; j < point_idxs_repeat.size(); ++j) {
                concatenated_points(i++) = point_idxs_repeat(j);
            }
            point_idxs = concatenated_points;

            std::random_shuffle(point_idxs.data(), point_idxs.data() + point_size);
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
            if (data_room.empty()) {
                data_room.push_back(data_batch.cast<float>()); // Ensure data_batch is also of type float
            } else {
                Eigen::Matrix<float, -1, -1> temp(data_room.back().rows() + data_batch.rows(), data_room.back().cols());
                temp << data_room.back(), data_batch;
                data_room.back() = temp;
            }
        }
    }

    
    // Get number of blocks from vector size
    int data_dim = 9; // Assuming each matrix in data_room_vector has 9 features per point

    // Create batch_data large enough to hold data from all matrices in the vector if needed
    Eigen::MatrixXf batch_data(data_room.size(), block_points * data_dim);
    Eigen::VectorXf batch_point_index(block_points);

    // Iterate over each matrix in the vector
    for (int idx = 0; idx < data_room.size(); ++idx) {
        const Eigen::MatrixXf& data_room_matrix = data_room[idx];

        // Verify the matrix has enough rows to extract a block of size 'block_points'
        if (data_room_matrix.rows() >= block_points) {
            // Extract the block from current matrix
            Eigen::MatrixXf data_block = data_room_matrix.block(0, 0, block_points, data_dim); // Adjust indices as needed
            batch_data.block(idx, 0, 1, block_points * data_dim) = Eigen::RowVectorXf::Map(data_block.data(), block_points * data_dim);
        }
        
    }

    // Transposing the data for any further operations if necessary
    Eigen::MatrixXf transposed_batch_data = batch_data.transpose();

    // Return as numpy array
    return py::array_t<float>(transposed_batch_data.size(), transposed_batch_data.data());

}




PYBIND11_MODULE(transform_to_pointnet, m) {
    m.doc() = "converts pointcloud data to pointnet features"; // optional module docstring

    m.def("cloud_to_pointnet", &transform_to_pointnet, "converts pointcloud data to pointnet features");
}