cmake_minimum_required(VERSION 3.16)

project(conc3ept)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set(CMAKE_CXX_STANDARD 17)

add_executable(rerankerserver ../server.cpp ../reranker.cpp)

add_library(asio_lib INTERFACE)
target_compile_definitions(rerankerserver PRIVATE ASIO_STANDALONE)

set_target_properties(rerankerserver PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(rerankerserver PRIVATE 
    ${CMAKE_SOURCE_DIR}/../../llama.cpp/include
    ${CMAKE_SOURCE_DIR}/../../llama.cpp/ggml/include
    ${CMAKE_SOURCE_DIR}/../../safetensors-cpp
)

target_include_directories(asio_lib INTERFACE "${CMAKE_SOURCE_DIR}/../../asio-1.36.0/include")

target_link_directories(rerankerserver PRIVATE 
    ${CMAKE_SOURCE_DIR}/../../llama.cpp/build/bin
)

target_link_libraries(rerankerserver PRIVATE llama asio_lib)