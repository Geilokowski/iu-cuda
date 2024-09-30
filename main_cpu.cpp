#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include "constants.h"
#include "utils/time_utils.h"
#include "utils/file_utils.h"

#include <Eigen/Dense>


RUNTIME_INFO linear_regression(const std::vector<std::vector<double> > &X, const std::vector<double> &Y) {
    std::chrono::steady_clock::time_point start_load = std::chrono::steady_clock::now();
    // Perform the OLS regression
    int rows = X.size();
    int cols = X[0].size();

    // Convert the vector of vectors X into an Eigen matrix
    Eigen::MatrixXd X_matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            X_matrix(i, j) = X[i][j];
        }
    }

    // Convert the vector Y into an Eigen vector
    Eigen::VectorXd Y_vector(rows);
    for (int i = 0; i < rows; ++i) {
        Y_vector(i) = Y[i];
    }

    // Add a column of ones to X_matrix for the intercept term
    Eigen::MatrixXd X_ext(rows, cols + 1);
    X_ext << Eigen::MatrixXd::Ones(rows, 1), X_matrix;

    // Perform the Ordinary Least Squares: (X'X)^-1 X'Y
    std::chrono::steady_clock::time_point begin_solve = std::chrono::steady_clock::now();
    Eigen::VectorXd beta = (X_ext.transpose() * X_ext).inverse() * X_ext.transpose() * Y_vector;

    std::cout << beta << std::endl;

    std::chrono::steady_clock::time_point end_solve = std::chrono::steady_clock::now();
    long long ms_load = std::chrono::duration_cast<std::chrono::milliseconds>(begin_solve - start_load).count();
    long long ms_calculate = std::chrono::duration_cast<std::chrono::milliseconds>(end_solve - begin_solve).count();

    std::cout << "Loaded data in " << ms_load << "ms, solved regression in " << ms_calculate << " ms" << std::endl;

    const RUNTIME_INFO rtInfo = {
        .load_time = ms_load,
        .calculate_time = ms_calculate,
    };

    return rtInfo;
}

int main() {
    std::cout << "Reading CSV file...";
    std::chrono::steady_clock::time_point begin_read = std::chrono::steady_clock::now();
    std::vector<double> Y;
    std::vector<std::vector<double> > X = readCSV(
        NYC_DATASET_PATH,
        Y);

    std::chrono::steady_clock::time_point end_read = std::chrono::steady_clock::now();
    std::cout << "Read CSV file in " << std::chrono::duration_cast<std::chrono::seconds>(
        end_read - begin_read).count() << " seconds" << std::endl;

    long long total_load_time = 0;
    long long total_calculate_time = 0;

    for(int i = 0; i < RUN_COUNT; i++) {
        RUNTIME_INFO timing = linear_regression(X, Y);
        total_load_time += timing.load_time;
        total_calculate_time += timing.calculate_time;
    }

    std::cout << "Average load time: " << total_load_time / RUN_COUNT << std::endl;
    std::cout << "Average calculate time: " << total_calculate_time / RUN_COUNT << std::endl;

    return 0;
}
