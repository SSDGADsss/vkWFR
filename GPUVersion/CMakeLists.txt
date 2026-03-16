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
find_package(Vulkan REQUIRED)
find_package(glslang REQUIRED)

set (LIB_TYPE SHARED)
string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)
find_package (HDF5 NAMES hdf5 COMPONENTS CXX ${SEARCH_TYPE})
set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIR}")
set (H5PL_LINK_LIBS ${H5PL_LINK_LIBS} ${HDF5_CXX_${LIB_TYPE}_LIBRARY})

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
vulkan_compile_shader(
    INFILE "shader/step5.comp"
    OUTFILE "shader/step5.hpp"
    NAMESPACE "comp")
vulkan_compile_shader(
    INFILE "shader/step6.comp"
    OUTFILE "shader/step6.hpp"
    NAMESPACE "comp")
vulkan_compile_shader(
    INFILE "shader/step7.comp"
    OUTFILE "shader/step7.hpp"
    NAMESPACE "comp")
vulkan_compile_shader(
    INFILE "shader/step8.comp"
    OUTFILE "shader/step8.hpp"
    NAMESPACE "comp")

add_library(shader INTERFACE
  "shader/step1.hpp"
  "shader/step2.hpp"
  "shader/step3.hpp"
  "shader/step4.hpp"
  "shader/step5.hpp"
  "shader/step6.hpp"
  "shader/step7.hpp"
  "shader/step8.hpp"
)

target_include_directories(shader
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

add_library(VkFFT INTERFACE)
target_compile_definitions(VkFFT INTERFACE VKFFT_BACKEND=0)

aux_source_directory(src SOURCE_DIR)

include_directories(
  inc
  ${GLSLANG_HEADER_PATH}
)
link_libraries(
  shader
  kompute::kompute
  SPIRV
  glslang
  Vulkan::Vulkan
  VkFFT
  ${H5PL_LINK_LIBS}
)
add_compile_definitions(VK_API_VERSION=13)
add_executable(${PROJECT_NAME} run/main.cpp ${SOURCE_DIR})
add_executable(largeImage run/largeImage.cpp ${SOURCE_DIR})
add_executable(smallImage run/smallImage.cpp ${SOURCE_DIR})
add_executable(GPUBenchmark run/benchmark.cpp ${SOURCE_DIR})

# 在benchmark生成后，将`../matlab/Phase`文件夹拷贝到构建目录中
add_custom_command(TARGET GPUBenchmark POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/matlab/Phase
        ${CMAKE_CURRENT_BINARY_DIR}/Phase
    COMMENT "Copying Phase folder to build directory for benchmark"
    VERBATIM
)
