cmake_minimum_required(VERSION 3.16)
project(rerankerserver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# src/reranker -> repo root
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

set(ASIO_INCLUDE "${REPO_ROOT}/external/asio/asio/include")
set(LLAMA_DIR    "${REPO_ROOT}/external/llama.cpp")
set(SAFETENSORS_INCLUDE "${REPO_ROOT}/external/safetensors-cpp")

find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

# Build llama.cpp as part of this project (second arg = its binary dir under build/)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ON CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_DIR}" llama_build)

add_executable(rerankerserver
    server.cpp
    reranker.cpp
    safetensors_impl.cpp
)

target_include_directories(rerankerserver PRIVATE
    ${ASIO_INCLUDE}
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
)
target_compile_definitions(rerankerserver PRIVATE ASIO_STANDALONE)

target_link_libraries(rerankerserver PRIVATE
    llama ggml
    OpenSSL::SSL OpenSSL::Crypto Threads::Threads)

set_target_properties(rerankerserver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")