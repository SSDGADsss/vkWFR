cmake_minimum_required(VERSION 3.26)

project(vkWFR-CPU)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_file(STB_HEADER stb/stb_image.h REQUIRED)
find_package(FFTW3 REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)

if(${CMAKE_BUILD_TYPE} MATCHES "Release")
  add_compile_options(-O3 -Wall)
else()
  add_compile_options(-O0 -g)
endif()

# CPUVersion
aux_source_directory(src SOURCE_DIR)

add_executable(${PROJECT_NAME} run/main.cpp ${SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC inc)
target_link_libraries(${PROJECT_NAME} ${FFTW3_LIBRARIES} Eigen3::Eigen)

