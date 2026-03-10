cmake_minimum_required(VERSION 3.26)

project(vkWFR-GPU)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_file(STB_HEADER stb/stb_image.h REQUIRED)
find_file(VKFFT_HEADER vkFFT.h REQUIRED)
find_path(GLSLANG_HEADER_PATH glslang_c_interface.h
HINTS
  /usr/include/glslang/Include/
  /usr/local/include/glslang/Include/
REQUIRED)
find_package(FFTW3 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Vulkan REQUIRED)
find_package(glslang REQUIRED)

option(KOMPUTE_OPT_GIT_TAG "The tag of the repo to use for the example" v0.9.0)

set(KOMPUTE_OPT_LOG_LEVEL Warn)

include(FetchContent)
fetchcontent_declare(kompute
  GIT_REPOSITORY https://github.com/KomputeProject/kompute.git
  GIT_TAG ${KOMPUTE_OPT_GIT_TAG})
fetchcontent_makeavailable(kompute)
include_directories(${kompute_SOURCE_DIR}/src/include)

if(${CMAKE_BUILD_TYPE} MATCHES "Release")
  add_compile_options(-O3 -Wall)
else()
  add_compile_options(-O0 -g)
endif()

vulkan_compile_shader(
    INFILE "shader/step1.comp"
    OUTFILE "shader/step1.hpp"
    NAMESPACE "comp")
vulkan_compile_shader(
    INFILE "shader/step2.comp"
    OUTFILE "shader/step2.hpp"
    NAMESPACE "comp")
vulkan_compile_shader(
    INFILE "shader/step3.comp"
    OUTFILE "shader/step3.hpp"
    NAMESPACE "comp")
vulkan_compile_shader(
    INFILE "shader/step4.comp"
    OUTFILE "shader/step4.hpp"
    NAMESPACE "comp")

add_library(shader INTERFACE
  "shader/step1.hpp"
  "shader/step2.hpp"
  "shader/step3.hpp"
  "shader/step4.hpp"
)
target_include_directories(shader
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

add_library(VkFFT INTERFACE)
target_compile_definitions(VkFFT INTERFACE
  -DVKFFT_BACKEND=0)

aux_source_directory(src SOURCE_DIR)

add_executable(${PROJECT_NAME} run/main.cpp ${SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC
  inc
  ${GLSLANG_HEADER_PATH}
)
target_link_libraries(${PROJECT_NAME} PUBLIC
  ${FFTW3_LIBRARIES}
  Eigen3::Eigen
  shader
  kompute::kompute
  SPIRV
  glslang
  Vulkan::Vulkan
  VkFFT
)
target_compile_definitions(${PROJECT_NAME}
  PUBLIC -DVK_API_VERSION=13)
