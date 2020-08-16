#define EIGEN_USE_MKL_ALL

#include <iostream>
#include "eigen/Eigen/Dense"

int main() {
    Eigen::MatrixXf m1;
    m1 << 1, 2, 3, 4;
    std::cout << m1 << std::endl;

    return 0;
}