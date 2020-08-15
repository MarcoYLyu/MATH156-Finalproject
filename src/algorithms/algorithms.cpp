// #define EIGEN_USE_MKL_VML

#include <iostream>
#include <iterator>
#include <algorithm>
#include "eigen/Eigen/Core"
#include "eigen/Eigen/SVD"

#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"

using namespace Eigen;
using namespace std;
namespace py=pybind11;

MatrixXf svd_impute(MatrixXf &data, int k = 0, int num_iter = 5) {
    int n = data.rows();
    int m = data.cols();
    if (k == 0) {
        k = std::min(n, m);
    }

    MatrixXf miss_matrix = MatrixXf::Constant(n, m, 0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (data(i, j) == 0) {
                miss_matrix(i, j) = 1;
            }
        }
    }
    MatrixXf imputed_matrix = data;

    for (int itr = 0; itr < num_iter; ++itr) {
        VectorXf mean = imputed_matrix.colwise().mean();
        MatrixXf temp = MatrixXf::Constant(n, m, 0);
        for (int i = 0; i < n; ++i) {
            temp.row(i) = miss_matrix.row(i).array() * mean.transpose().array();
        }
        JacobiSVD<MatrixXf> svd(temp, ComputeThinU | ComputeThinV);
        MatrixXf approx_matrix = svd.matrixU().leftCols(k) * svd.singularValues().head(k).asDiagonal() * svd.matrixV().leftCols(k).transpose();
        imputed_matrix = data + miss_matrix.cwiseProduct(approx_matrix);
    }

    return imputed_matrix;
}

using namespace pybind11::literals;
PYBIND11_MODULE(algorithms, m) {
    m.def("svd_impute", &svd_impute, "Impute the matrix with SVD", py::arg("data"), py::arg("k")=0, py::arg("num_iter")=1);
}

