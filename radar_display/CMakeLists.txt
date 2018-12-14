cmake_minimum_required(VERSION 2.8.3)
project(radar_display)

add_definitions(-std=c++11 -g -O3 -ftree-vectorize -march=native)

find_package(catkin REQUIRED COMPONENTS roscpp cv_bridge message_filters sensor_msgs image_transport)
find_package(OpenCV REQUIRED)

catkin_package(
)

include_directories(
${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
)

add_executable(radar src/radar.cpp)
target_link_libraries(radar ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(global_path src/global_path.cpp)
target_link_libraries(global_path ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(lantern src/lantern.cpp)
target_link_libraries(lantern  ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(lantern_with_pannel src/lantern_with_pannel.cpp)
target_link_libraries(lantern_with_pannel  ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
