cmake_minimum_required(VERSION 2.8.3)
project(radar_display)


find_package(catkin REQUIRED COMPONENTS roscpp cv_bridge)
find_package(OpenCV REQUIRED)

catkin_package(
)

include_directories(
${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
)

add_executable(radar src/radar.cpp)
target_link_libraries(radar ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})


