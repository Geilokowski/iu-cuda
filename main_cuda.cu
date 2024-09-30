#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "constants.h"
#include "utils/time_utils.h"
#include "utils/file_utils.h"

// Error checking macros
#define CUDA_CHECK(err) { gpuAssert((err), __FILE__, __LINE__); }
#define CUBLAS_CHECK(err) { cublasAssert((err), __FILE__, __LINE__); }
#define CUSOLVER_CHECK(err) { cusolverAssert((err), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                << " in " << file << " at line " << line << std::endl;
        exit(code);
    }
}

inline void cublasAssert(cublasStatus_t code, const char *file, int line) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS Error in " << file << " at line " << line << std::endl;
        exit(code);
    }
}

inline void cusolverAssert(cusolverStatus_t code, const char *file, int line) {
    if (code != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSolver Error in " << file << " at line " << line << std::endl;
        exit(code);
    }
}

RUNTIME_INFO linear_regression(const std::vector<std::vector<double> > &X, const std::vector<double> &Y) {
    std::chrono::steady_clock::time_point start_load = std::chrono::steady_clock::now();

    int N = Y.size(); // Number of samples
    int M = X[0].size(); // Number of features

    // Add intercept term and prepare data in column-major order
    size_t numFeatures = M + 1; // Include intercept
    std::vector<double> h_X_col_major(numFeatures * N);

    // First column (intercept term)
    for (size_t i = 0; i < N; ++i) {
        h_X_col_major[i] = 1.0;
    }

    // Remaining features
    for (size_t j = 0; j < M; ++j) {
        for (size_t i = 0; i < N; ++i) {
            h_X_col_major[(j + 1) * N + i] = X[i][j];
        }
    }

    // Prepare Y
    std::vector<double> h_Y = Y;

    // Allocate device memory
    double *d_X = nullptr, *d_Y = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_X, sizeof(double) * numFeatures * N));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, sizeof(double) * N));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X_col_major.data(), sizeof(double) * numFeatures * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y.data(), sizeof(double) * N, cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t cublasHandle;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));

    // Create cuSolver handle
    cusolverDnHandle_t cusolverHandle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverHandle));

    std::chrono::steady_clock::time_point begin_solve = std::chrono::steady_clock::now();

    double alpha = 1.0;
    double beta = 0.0;

    // Compute X^T X
    double *d_XtX = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_XtX, sizeof(double) * numFeatures * numFeatures));
    CUBLAS_CHECK(cublasDgemm(
        cublasHandle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        numFeatures,
        numFeatures,
        N,
        &alpha,
        d_X, N,
        d_X, N,
        &beta,
        d_XtX, numFeatures));

    // Compute X^T Y
    double *d_XtY = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_XtY, sizeof(double) * numFeatures));
    CUBLAS_CHECK(cublasDgemv(
        cublasHandle,
        CUBLAS_OP_T,
        N,
        numFeatures,
        &alpha,
        d_X, N,
        d_Y, 1,
        &beta,
        d_XtY, 1));

    // Allocate workspace and info
    int work_size = 0;
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolverHandle, numFeatures, numFeatures, d_XtX, numFeatures, &work_size));

    double *d_work = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_work, sizeof(double) * work_size));

    int *d_Ipiv = nullptr, *d_info = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_Ipiv, sizeof(int) * numFeatures));
    CUDA_CHECK(cudaMalloc((void**)&d_info, sizeof(int)));

    // LU factorization
    CUSOLVER_CHECK(
        cusolverDnDgetrf(
            cusolverHandle,
            numFeatures,
            numFeatures,
            d_XtX,
            numFeatures,
            d_work,
            d_Ipiv,
            d_info));

    // Check if factorization was successful
    int info_host = 0;
    CUDA_CHECK(cudaMemcpy(&info_host, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_host != 0) {
        std::cerr << "LU factorization failed with info = " << info_host << std::endl;
        exit(1);
    }

    // Solve the system
    CUSOLVER_CHECK(
        cusolverDnDgetrs(
            cusolverHandle,
            CUBLAS_OP_N,
            numFeatures,
            1,
            d_XtX,
            numFeatures,
            d_Ipiv,
            d_XtY,
            numFeatures,
            d_info));

    // Check if solve was successful
    CUDA_CHECK(cudaMemcpy(&info_host, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_host != 0) {
        std::cerr << "Solving linear system failed with info = " << info_host << std::endl;
        exit(1);
    }

    // Copy the result back to host
    std::vector<double> h_beta(numFeatures);
    CUDA_CHECK(cudaMemcpy(h_beta.data(), d_XtY, sizeof(double) * numFeatures, cudaMemcpyDeviceToHost));

    // Print the coefficients
    std::cout << "Regression coefficients:" << std::endl;
    for (size_t i = 0; i < numFeatures; ++i) {
        std::cout << "beta[" << i << "] = " << h_beta[i] << std::endl;
    }

    std::chrono::steady_clock::time_point end_solve = std::chrono::steady_clock::now();
    long long ms_load = std::chrono::duration_cast<std::chrono::milliseconds>(begin_solve - start_load).count();
    long long ms_calculate = std::chrono::duration_cast<std::chrono::milliseconds>(end_solve - begin_solve).count();

    std::cout << "Loaded data in " << ms_load << "ms, solved regression in " << ms_calculate << " ms" << std::endl;

    CUDA_CHECK(cudaFree(d_XtX));
    CUDA_CHECK(cudaFree(d_XtY));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_info));

    CUBLAS_CHECK(cublasDestroy(cublasHandle));

    // Cleanup
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverHandle));
    CUDA_CHECK(cudaDeviceReset());

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
