cmake_minimum_required(VERSION 3.10)
project(yolo_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
include_directories(
    /usr/local/src/tensorflow
    /usr/local/src/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include
    /usr/local/src/tensorflow/tensorflow/lite/tools/make/downloads/absl
    ${OpenCV_INCLUDE_DIRS}  
)

link_directories(
    /usr/local/src/tensorflow/tensorflow/lite/tools/make/gen/linux_aarch64/lib
)


include_directories(
    ai/architectures
    ai/inferencers
    ai/processors
    detection
    image
    model
)

set(SOURCE_FILES 
    ai/architectures/ultralyticsyolo.cpp
    ai/inferencers/litert.cpp
    ai/inferencers/onnxrt.cpp
    ai/processors/detector.cpp
    detection/postprocessing.cpp
    image/preprocessing.cpp
    model/detection.cpp
    main.cpp
)

set(EXECUTABLE
    yolo_benchmark
)

add_executable(${EXECUTABLE} ${SOURCE_FILES})

target_link_libraries(${EXECUTABLE} PRIVATE 
    ${OpenCV_LIBS}
    onnxruntime 
    tensorflow-lite
    flatbuffers
    dl
    Threads::Threads
    -lstdc++fs
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fexceptions -pthread")
    