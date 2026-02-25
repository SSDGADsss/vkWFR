cmake_minimum_required(VERSION 3.26)

project(vkWFR-GPU)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_file(STB_HEADER stb/stb_image.h REQUIRED)
find_package(FFTW3 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

option(KOMPUTE_OPT_GIT_TAG "The tag of the repo to use for the example" v0.9.0)

include(FetchContent)
FetchContent_Declare(kompute GIT_REPOSITORY https://github.com/KomputeProject/kompute.git
        GIT_TAG ${KOMPUTE_OPT_GIT_TAG})
FetchContent_MakeAvailable(kompute)
include_directories(${kompute_SOURCE_DIR}/src/include)

if(${CMAKE_BUILD_TYPE} MATCHES "Release")
  add_compile_options(-O3 -Wall)
else()
  add_compile_options(-O0 -g)
endif()

vulkan_compile_shader(
    INFILE shader/test.comp
    OUTFILE shader/test.hpp
    NAMESPACE "shader")

add_library(shader INTERFACE "shader/test.hpp")
target_include_directories(shader INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

add_executable(${PROJECT_NAME} run/main.cpp ${SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC inc)
target_link_libraries(${PROJECT_NAME} PUBLIC
  ${FFTW3_LIBRARIES}
  Eigen3::Eigen
  shader
  kompute::kompute
)

