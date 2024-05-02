#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <pybind11/pybind11.h>

namespace py = pybind11;

Eigen::MatrixXf transform_to_pointnet(const Eigen::MatrixXf& clouddata) {
    // parameters
    float block_size = 1.0f;
    float stride = 0.5f;
    int block_points = 4096;
    float padding = 0.001f;

    // transform to Eigen
    Eigen::MatrixXf scenepointsx(clouddata.rows(), 4); // TODO: what is this 4?
    scenepointsx.leftCols(3) = clouddata.leftCols(3);
    scenepointsx.col(3).setOnes();
    Eigen::RowVector3f coord_min = scenepointsx.colwise().minCoeff();
    Eigen::RowVector3f coord_max = scenepointsx.colwise().maxCoeff();
    Eigen::VectorXf labelweights = Eigen::VectorXf::Ones(13);
    Eigen::MatrixXf point_set_ini = scenepointsx;
    Eigen::MatrixXf points = point_set_ini;
    coord_min = points.leftCols(3).colwise().minCoeff();
    coord_max = points.leftCols(3).colwise().maxCoeff();
    int grid_x = static_cast<int>(std::ceil((coord_max(0) - coord_min(0) - block_size) / stride)) + 1;
    int grid_y = static_cast<int>(std::ceil((coord_max(1) - coord_min(1) - block_size) / stride)) + 1;
    Eigen::MatrixXf data_room(0, 9);
    Eigen::MatrixXi index_room(0, block_points);

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
            Eigen::Array<bool, Eigen::Dynamic, 1> point_idxs = (points.col(0).array() >= s_x - padding) && (points.col(0).array() <= e_x + padding) && (points.col(1).array() >= s_y - padding) && (points.col(1).array() <= e_y + padding); // TODO: validate this
            if (point_idxs.count() == 0) {
                continue;
            }
            int num_batch = static_cast<int>(std::ceil(point_idxs.count() / static_cast<float>(block_points)));
            int point_size = num_batch * block_points;
            bool replace = (point_size - point_idxs.count() <= point_idxs.count()) ? false : true;
            Eigen::VectorXi point_idxs_repeat = Eigen::VectorXi::NullaryExpr(point_size - point_idxs.count(), [&]{ return gen() % point_idxs.count(); });
            Eigen::VectorXi point_idxs_all = Eigen::VectorXi::Zero(point_size);
            point_idxs_all.head(point_idxs.count()) = Eigen::VectorXi(points.rows()).setLinSpaced(points.rows()).array()[point_idxs.matrix()];
            point_idxs_all.tail(point_size - point_idxs.count()) = point_idxs_repeat;
            std::shuffle(point_idxs_all.data(), point_idxs_all.data() + point_size, gen);
            Eigen::MatrixXf data_batch = points.row(point_idxs_all.transpose());
            Eigen::MatrixXf normlized_xyz(point_size, 3);
            normlized_xyz.col(0) = data_batch.col(0) / coord_max(0);
            normlized_xyz.col(1) = data_batch.col(1) / coord_max(1);
            normlized_xyz.col(2) = data_batch.col(2) / coord_max(2);
            data_batch.col(0) -= s_x + block_size / 2.0f;
            data_batch.col(1) -= s_y + block_size / 2.0f;
            data_batch.rightCols(3) /= 255.0f;
            data_batch.rightCols(3) = normlized_xyz;
            data_room.conservativeResize(data_room.rows() + data_batch.rows(), data_batch.cols());
            data_room.bottomRows(data_batch.rows()) = data_batch;
            index_room.conservativeResize(index_room.rows() + point_idxs_all.size(), point_idxs_all.size());
            index_room.bottomRows(point_idxs_all.size()) = point_idxs_all.transpose();
        }
    }

    data_room.conservativeResize(data_room.rows(), 9);
    index_room.conservativeResize(index_room.rows(), block_points);

    return data_room.transpose();
}

PYBIND11_MODULE(cloud2pointnet, m) {
    m.doc() = "converts pointcloud data to pointnet features"; // optional module docstring

    m.def("transform_to_pointnet", &transform_to_pointnet, "converts pointcloud data to pointnet features");
}