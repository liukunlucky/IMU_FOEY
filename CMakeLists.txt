cmake_minimum_required(VERSION 3.12)
project(test_IMU_VP)

set(CMAKE_CXX_STANDARD 14)

include_directories( "/usr/local/include/eigen3" )
find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(test_IMU_VP test_imu_foey_vp.cpp)
target_link_libraries( test_IMU_VP ${OpenCV_LIBS} )