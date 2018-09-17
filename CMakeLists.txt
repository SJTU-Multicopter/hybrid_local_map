cmake_minimum_required(VERSION 3.10)
project(hybrid_local_map)

set(CMAKE_CXX_STANDARD 11)

include_directories(ewok_ring_buffer/include)
include_directories(ewok_ring_buffer/include/ewok)
include_directories(yolo2_ncs/src)

add_executable(hybrid_local_map
        ewok_ring_buffer/include/ewok/ed_nor_ring_buffer.h
        ewok_ring_buffer/include/ewok/ed_ring_buffer.h
        ewok_ring_buffer/include/ewok/raycast_ring_buffer.h
        ewok_ring_buffer/include/ewok/ring_buffer_base.h
        ewok_ring_buffer/src/realtime_example.cpp
        ewok_ring_buffer/src/ring_buffer_example.cpp
        ewok_ring_buffer/src/tum_rgbd_ring_buffer_example.cpp
        ewok_ring_buffer/test/ring-buffer-base-test.cpp
        ewok_ring_buffer/CMakeLists.txt
        secure_flying/src/mapping.cpp
        secure_flying/src/sim_flying.cpp
        secure_flying/CMakeLists.txt
        yolo2_ncs/src/Common.cpp
        yolo2_ncs/src/Common.h
        yolo2_ncs/src/PythonWrapper.cpp
        yolo2_ncs/src/Region.cpp
        yolo2_ncs/src/Region.h
        yolo2_ncs/CMakeLists.txt)
