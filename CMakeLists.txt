cmake_minimum_required(VERSION 3.10)
project(UAVDetector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(nlohmann_json REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}  # This is key - adds the root as include path
)

add_subdirectory(tools)
add_subdirectory(io)

add_executable(visualize visualize.cpp)

target_link_libraries(visualize 
    ${OpenCV_LIBS}
    fmt::fmt 
    yaml-cpp
    tools
    io
)

set_target_properties(visualize PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)