cmake_minimum_required(VERSION 3.26)

project(vkWFR-CPU)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(FFTW3 REQUIRED)
# CPUVersion
aux_source_directory(src SOURCE_DIR)

add_executable(${PROJECT_NAME} run/main.cpp ${SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC inc)
target_link_libraries(${PROJECT_NAME} ${FFTW3_LIBRARIES})

