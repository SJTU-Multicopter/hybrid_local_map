cmake_minimum_required(VERSION 2.8.3)
project(yolov3)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  #object_msg
)

catkin_package(

)

include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)


