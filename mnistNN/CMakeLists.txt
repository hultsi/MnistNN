cmake_minimum_required(VERSION 3.21)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(MnistNN VERSION 0.0.1 DESCRIPTION "Random MnistNN library")

# Bundle library
add_library(${PROJECT_NAME} STATIC
    src/mnistParser.cpp
    src/statpack.cpp
    src/train_v1.cpp
)

# TODO: change to PRIVATE and create public interface ?
# include headers
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/headers)