cmake_minimum_required(VERSION 3.21)

# specify compilers
# set(CMAKE_C_COMPILER gcc)
# set(CMAKE_CXX_COMPILER g++)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(MnistNN VERSION 0.0.1 DESCRIPTION "Random MnistNN library")

# set_target_properties(MnistNN PROPERTIES VERSION ${PROJECT_VERSION})

# Bundle library
add_library(${PROJECT_NAME} SHARED
    src/mnistParser.cpp
    src/statpack.cpp
    src/train_v1.cpp
)

# include headers
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/headers)