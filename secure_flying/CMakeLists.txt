cmake_minimum_required(VERSION 2.8.3)
project(secure_flying)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
add_definitions(-std=c++11 -g -O3 -ftree-vectorize -march=native)
# find_package(catkin_simple REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  message_filters
  tf
)

#find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(SuiteSparse REQUIRED)

add_definitions(${PCL_DEFINITIONS})
# catkin_simple()
catkin_package()

include_directories(${PCL_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(${EIGEN3_INCLUDE_DIR}) #${CHOLMOD_INCLUDE_DIR}#)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/../ewok_ring_buffer/include)
include_directories(${PROJECT_SOURCE_DIR}/../ewok_optimization/include)
include_directories(${PROJECT_SOURCE_DIR}/../ewok_poly_spline/include)

#add_executable(sim_flying src/sim_flying.cpp)
#target_link_libraries(sim_flying ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${CHOLMOD_LIBRARY} 
#/usr/local/lib/libnlopt.so.0.9.0)

#add_executable(mapping src/mapping.cpp)
#target_link_libraries(mapping ${PCL_LIBRARIES} ${catkin_LIBRARIES})
#
#add_executable(simple_mapping src/simple_mapping.cpp)
#target_link_libraries(simple_mapping ${PCL_LIBRARIES} ${catkin_LIBRARIES})

#add_executable(local_planning src/local_planning.cpp)
#target_link_libraries(local_planning ${PCL_LIBRARIES} ${catkin_LIBRARIES})

add_executable(local_planning_dynamic src/local_planning_dynamic.cpp)
target_link_libraries(local_planning_dynamic ${PCL_LIBRARIES} ${catkin_LIBRARIES})

add_executable(display_costs src/display_costs.cpp)
target_link_libraries(display_costs ${OpenCV_LIBS} ${catkin_LIBRARIES})

