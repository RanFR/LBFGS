cmake_minimum_required(VERSION 2.8.3)
project(lbfgs)

find_package(Eigen3 REQUIRED)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_origin src/lbfgs_example.cpp)

add_executable(${PROJECT_NAME}_exec src/test.cpp)
