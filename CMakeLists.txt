cmake_minimum_required(VERSION 3.26)

#set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

project(NeuralNetwork)

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}\\src)
link_directories(${OpenCL_LIBRARY})

message(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
message(OpenCL_INCLUDE_DIRS="${OpenCL_INCLUDE_DIRS}")
message(OpenCL_LIBRARY="${OpenCL_LIBRARY}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

set(EXECUTABLE_NAME NeuralNetwork)

add_executable(${EXECUTABLE_NAME} src/Entry.cpp)

target_sources(${EXECUTABLE_NAME} PRIVATE
src/Entry.cpp
src/reader/data_reader.h
src/reader/mnist_reader_common.hpp
)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_link_libraries(${EXECUTABLE_NAME} OpenCL::OpenCL)