cmake_minimum_required(VERSION 3.21)

# specify compilers
# set(CMAKE_C_COMPILER gcc)
# set(CMAKE_CXX_COMPILER g++)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set the project name
project(MnistNNTestApp VERSION 0.0.1 DESCRIPTION "Random MnistNN test app")

# add the executable
add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_LIST_DIR}/main.cpp)

# include library
target_link_libraries(${PROJECT_NAME} MnistNN)