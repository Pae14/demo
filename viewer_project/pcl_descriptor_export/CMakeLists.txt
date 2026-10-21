cmake_minimum_required(VERSION 3.12)
project(pcl_descriptor_export LANGUAGES C CXX)

find_package(PCL REQUIRED COMPONENTS common io features search)

add_executable(pcl_descriptor_export main.cpp)
target_compile_features(pcl_descriptor_export PRIVATE cxx_std_14)
target_include_directories(pcl_descriptor_export PRIVATE ${PCL_INCLUDE_DIRS})
target_link_directories(pcl_descriptor_export PRIVATE ${PCL_LIBRARY_DIRS})
target_compile_options(pcl_descriptor_export PRIVATE ${PCL_DEFINITIONS})
target_link_libraries(pcl_descriptor_export PRIVATE ${PCL_LIBRARIES})
