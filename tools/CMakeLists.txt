cmake_minimum_required(VERSION 3.16)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)  
add_library(tools OBJECT 
    exiter.cpp
    extended_kalman_filter.cpp
    ransac_sine_fitter.cpp
    img_tools.cpp
    math_tools.cpp
    plotter.cpp
    trajectory.cpp
    recorder.cpp
    logger.cpp
    pid.cpp
    crc.cpp
    targetyawpitch.cpp
    camera2gimbal.cpp
    solver.cpp
)
target_link_libraries(tools PUBLIC fmt::fmt spdlog::spdlog)
target_include_directories(tools PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})