cmake_minimum_required(VERSION 2.8.3)
project(secure_flying)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
add_definitions(-std=c++11 -g -O1 -ftree-vectorize -march=native)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  message_filters
  mav_msgs
  geometry_msgs
  sensor_msgs
  gazebo_msgs
  tf
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

add_definitions(${PCL_DEFINITIONS})

catkin_package()

include_directories(${PCL_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/../ewok_ring_buffer/include)
include_directories(${PROJECT_SOURCE_DIR}/../ewok_optimization/include)
include_directories(${PROJECT_SOURCE_DIR}/../ewok_poly_spline/include)


add_executable(local_planning_dynamic src/local_planning_dynamic.cpp)
target_link_libraries(local_planning_dynamic ${PCL_LIBRARIES} ${catkin_LIBRARIES})

add_executable(display_costs src/display_costs.cpp)
target_link_libraries(display_costs ${OpenCV_LIBS} ${catkin_LIBRARIES})

