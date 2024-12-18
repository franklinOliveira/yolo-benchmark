cmake_minimum_required(VERSION 3.10)

# Project name
project(ONNXInference)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the path to OpenCV (adjust if needed)
find_package(OpenCV REQUIRED)

# Add your source files here
add_executable(ONNXInference main.cpp)

# Link ONNX Runtime and OpenCV libraries
target_link_libraries(ONNXInference PRIVATE onnxruntime ${OpenCV_LIBS})

# Include directories for ONNX Runtime and OpenCV
target_include_directories(ONNXInference PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
