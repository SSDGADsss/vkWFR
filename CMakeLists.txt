cmake_minimum_required(VERSION 3.26)

project(vkWFR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(CMAKE_BUILD_TYPE "CMake Build Type (Debug,Release)" Debug)

add_subdirectory(CPUVersion)
add_subdirectory(GPUVersion)
