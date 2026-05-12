cmake_minimum_required(VERSION 3.10)
project(UAVDetector LANGUAGES CXX CUDA)

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

set(TENSORRT_PATH "/home/radar/TensorRT-8.6.1.6" CACHE PATH "Path to TensorRT")
include_directories(${TENSORRT_PATH}/include)
link_directories(${TENSORRT_PATH}/lib)

find_library(NVINFER_LIBRARY nvinfer PATHS ${TENSORRT_PATH}/lib REQUIRED)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin PATHS ${TENSORRT_PATH}/lib REQUIRED)
find_library(NVONNX_PARSER_LIB nvonnxparser PATHS ${TENSORRT_PATH}/lib REQUIRED)

set(TRTYOLO_DIR "/home/radar/TensorRT-YOLO" CACHE PATH "Path to TRTYOLO")
if(NOT EXISTS ${TRTYOLO_DIR})
    message(FATAL_ERROR "TRTYOLO directory not found: ${TRTYOLO_DIR}")
endif()
set(TRTYOLO_INCLUDE_DIR ${TRTYOLO_DIR}/include)
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
message(STATUS "TRTYOLO library: ${TRTYOLO_LIBRARY}")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)   # <-- added

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
add_executable(visualize visualize.cpp)

target_link_libraries(visualize
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

set_compile_options(visualize)
set_target_properties(visualize PROPERTIES
    CUDA_ARCHITECTURES "61;70;75;86"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)