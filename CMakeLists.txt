cmake_minimum_required(VERSION 3.29)
project(iu_cuda_assignment CUDA CXX)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

add_executable(
        iu_linear_regression_cuda
        main_cuda.cu
        utils/file_utils.cpp
        utils/file_utils.h
        constants.h
        utils/time_utils.h)

set_target_properties(
        iu_linear_regression_cuda
        PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set_target_properties(
        iu_linear_regression_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(
        iu_linear_regression_cuda
        CUDA::cublas
        CUDA::cusolver)


add_executable(
        iu_linear_regression_cpu
        main_cpu.cpp
        utils/file_utils.cpp
        utils/file_utils.h
        constants.h
        utils/time_utils.h)

target_include_directories(
        iu_linear_regression_cpu
        PRIVATE ${CMAKE_SOURCE_DIR}/Eigen)