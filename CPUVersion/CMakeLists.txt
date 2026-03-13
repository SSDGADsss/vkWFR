cmake_minimum_required(VERSION 3.26)

project(vkWFR-CPU)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_file(STB_HEADER stb/stb_image.h REQUIRED)
find_package(FFTW3 REQUIRED)

if(${CMAKE_BUILD_TYPE} MATCHES "Release")
  add_compile_options(-O3 -Wall)
else()
  add_compile_options(-O0 -g)
endif()

# CPUVersion
aux_source_directory(src SOURCE_DIR)
include_directories(inc)

add_executable(${PROJECT_NAME} run/main.cpp ${SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC inc)
target_link_libraries(${PROJECT_NAME} ${FFTW3_LIBRARIES})

add_executable(CPUBenchmark run/benchmark.cpp ${SOURCE_DIR})
target_include_directories(CPUBenchmark PUBLIC inc)
target_link_libraries(CPUBenchmark ${FFTW3_LIBRARIES})

# 在benchmark生成后，将`../matlab/Phase`文件夹拷贝到构建目录中
add_custom_command(TARGET CPUBenchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/matlab/Phase
        ${CMAKE_CURRENT_BINARY_DIR}/Phase
    COMMENT "Copying Phase folder to build directory for benchmark"
    VERBATIM
)
