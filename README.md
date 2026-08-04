cmake_minimum_required(VERSION 3.10)
project(sp_antidrone_26 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(MPI)
if(MPI_FOUND)
  if(NOT TARGET MPI::MPI_C)
    add_library(MPI::MPI_C INTERFACE IMPORTED)
  endif()
  if(NOT TARGET MPI::MPI_CXX)
    add_library(MPI::MPI_CXX INTERFACE IMPORTED)
  endif()
endif()

find_package(CURL REQUIRED)

# ------------------------------------------------------------------------------
# Dependencies (CUDA, TensorRT, TRTYOLO, OpenCV, etc.)
# ------------------------------------------------------------------------------
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
set(CUDA_ROOT "/usr/local/cuda")
include_directories(${CUDA_ROOT}/include)
link_directories(${CUDA_ROOT}/lib64)
find_library(CUDA_LIBRARY cudart HINTS ${CUDA_ROOT}/lib64)

set(TENSORRT_PATH "~/TensorRT-8.6.1.6" CACHE PATH "Path to TensorRT")
include_directories(${TENSORRT_PATH}/include)
link_directories(${TENSORRT_PATH}/lib)

find_library(NVINFER_LIBRARY nvinfer PATHS ${TENSORRT_PATH}/lib REQUIRED)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin PATHS ${TENSORRT_PATH}/lib REQUIRED)
find_library(NVONNX_PARSER_LIB nvonnxparser PATHS ${TENSORRT_PATH}/lib REQUIRED)

set(TRTYOLO_DIR "${CMAKE_SOURCE_DIR}/TensorRT-YOLO" CACHE PATH "Path to TRTYOLO")
if(NOT EXISTS ${TRTYOLO_DIR})
    message(FATAL_ERROR "TRTYOLO directory not found: ${TRTYOLO_DIR}")
endif()
set(TRTYOLO_INCLUDE_DIR ${TRTYOLO_DIR}/install/include)
if(NOT EXISTS ${TRTYOLO_INCLUDE_DIR})
    set(TRTYOLO_INCLUDE_DIR ${TRTYOLO_DIR}/include)
endif()
if(NOT EXISTS ${TRTYOLO_INCLUDE_DIR})
    set(TRTYOLO_INCLUDE_DIR ${TRTYOLO_DIR}/modules/trtyolo/include)
endif()
set(TRTYOLO_LIBRARY_DIR ${TRTYOLO_DIR}/lib)
if(NOT EXISTS ${TRTYOLO_LIBRARY_DIR})
    set(TRTYOLO_LIBRARY_DIR ${TRTYOLO_DIR}/build)
endif()
find_library(TRTYOLO_LIBRARY
    NAMES trtyolo
    PATHS ${TRTYOLO_LIBRARY_DIR}
          ${TRTYOLO_DIR}/build/modules/trtyolo
          ${TRTYOLO_DIR}/modules/trtyolo
    REQUIRED
)
# Extract the directory containing libtrtyolo.so for RPATH
get_filename_component(TRTYOLO_RPATH_DIR "${TRTYOLO_LIBRARY}" DIRECTORY)
# Also add the install/lib path if it exists
set(TRTYOLO_INSTALL_LIB_DIR "${TRTYOLO_DIR}/install/lib")
message(STATUS "TRTYOLO library: ${TRTYOLO_LIBRARY}")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)   # <-- added

# ---- ROS 2 ----
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(radar_msgs REQUIRED)

# ------------------------------------------------------------------------------
# Include directories
# ------------------------------------------------------------------------------
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${TRTYOLO_INCLUDE_DIR}
    ${TENSORRT_PATH}/include
    ${CUDA_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tools
)

# ------------------------------------------------------------------------------
# Compiler optimisation helper
# ------------------------------------------------------------------------------
function(set_compile_options target)
    if(MSVC)
        target_compile_options(${target} PUBLIC $<$<CONFIG:Release>:-O2>)
        set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        target_compile_options(${target} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-O3 -flto=auto>)
        target_link_options(${target} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-O3 -flto=auto>)
    endif()
endfunction()

# ------------------------------------------------------------------------------
# Subdirectories (tools and io)
# ------------------------------------------------------------------------------
add_subdirectory(tools)
add_subdirectory(io)

# ------------------------------------------------------------------------------
# Executables
# ------------------------------------------------------------------------------
add_executable(antidrone
    src/main.cpp
    src/antidrone_node.cpp
    src/tracker.cpp
    src/detectors/base_detector.cpp
    src/detectors/light_bar_detector.cpp
    src/detectors/yolo_detector.cpp
    src/detectors/detector.cpp
)

target_link_libraries(antidrone
    ${OpenCV_LIBS}
    yaml-cpp
    tools
    io
    ${TRTYOLO_LIBRARY}
    ${NVINFER_LIBRARY}
    ${NVINFER_PLUGIN_LIBRARY}
    ${NVONNX_PARSER_LIB}
    ${CUDA_LIBRARY}
    ${CURL_LIBRARIES}
    fmt::fmt                 # explicit
    spdlog::spdlog           # explicit
)

set_compile_options(antidrone)
set_target_properties(antidrone PROPERTIES
    CUDA_ARCHITECTURES "61;70;75;86"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${TRTYOLO_RPATH_DIR};${TRTYOLO_INSTALL_LIB_DIR};${TENSORRT_PATH}/lib;${CUDA_ROOT}/lib64;${CMAKE_CURRENT_SOURCE_DIR}/io/mindvision/lib/amd64;${CMAKE_CURRENT_SOURCE_DIR}/io/hikrobot/lib/amd64"
    INSTALL_RPATH "${TRTYOLO_RPATH_DIR};${TRTYOLO_INSTALL_LIB_DIR};${TENSORRT_PATH}/lib;${CUDA_ROOT}/lib64;${CMAKE_CURRENT_SOURCE_DIR}/io/mindvision/lib/amd64;${CMAKE_CURRENT_SOURCE_DIR}/io/hikrobot/lib/amd64"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

ament_target_dependencies(antidrone
    rclcpp
    radar_msgs
)

install(TARGETS
    antidrone
    DESTINATION lib/${PROJECT_NAME}
)

ament_package()

