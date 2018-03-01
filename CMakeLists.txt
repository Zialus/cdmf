# Minimal version of CMake
cmake_minimum_required (VERSION 3.0)

if(UNIX AND NOT APPLE)
    message("Importing alternative FindOpenCL for Linux")
    set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
endif(UNIX AND NOT APPLE)

# Define project name
project(cdmf)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_C_STANDARD 11)

#set(CMAKE_VERBOSE_MAKEFILE ON)

# V1: native, V2: thread batching, V3: load balancing
set(VERSION "V3")
set(DFLAGS "-DNUM_RUN=1")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${DFLAGS} -D ${VERSION} -D VALUE_TYPE=double -m64 ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DFLAGS} -D ${VERSION} -D VALUE_TYPE=double -m64 ${OpenMP_CXX_FLAGS}")


# Source code of application
file(GLOB source_src "${PROJECT_SOURCE_DIR}/*.h")
file(GLOB source_hdr "${PROJECT_SOURCE_DIR}/*.cpp")

# Compiler flags
#if (CMAKE_COMPILER_IS_GNUCC)
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
#endif (CMAKE_COMPILER_IS_GNUCC)


# Set up executable
include_directories(${OpenCL_INCLUDE_DIR})
add_executable(cdmf ${source_src} ${source_hdr})
target_link_libraries(cdmf ${OpenCL_LIBRARIES} ${OMP_LIBRARIES})