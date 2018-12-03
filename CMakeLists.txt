cmake_minimum_required(VERSION 3.12)
project(gaussian_filter)

set(CMAKE_CXX_STANDARD 14)

enable_language(CUDA)

set(SOURCE_FILES gf.cu)
add_executable(gaussian_filter gf.cu)