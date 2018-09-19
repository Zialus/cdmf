# Minimal version of CMake
cmake_minimum_required (VERSION 3.9)

# Define project name
project(cdmf)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 14)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/exec/")

#set(CMAKE_VERBOSE_MAKEFILE ON)

#set(WARNINGS "-Wno-deprecated-declarations -Wno-format")
set(VALUE "-DVALUE_TYPE=double")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${VALUE} -m64 ${WARNINGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VALUE} -m64 ${WARNINGS}")


# Source code of application
file(GLOB source_src "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB source_hdr "${PROJECT_SOURCE_DIR}/src/*.h" "${PROJECT_SOURCE_DIR}/src/detail/*.h" "${PROJECT_SOURCE_DIR}/src/detail/opencl/*.h")

# Compiler flags
#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
#elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
#endif()


# Set up executable
add_executable(cdmf ${source_src} ${source_hdr})
target_link_libraries(cdmf OpenCL::OpenCL OpenMP::OpenMP_CXX)

enable_testing()
add_test(NAME a COMMAND $<TARGET_FILE:cdmf> -k 10 -t 5 -T 3 -l 0.05 -r 1 -q 1 -P 1 -V $ENV{VERSION_FLAG} -nThreadsPerBlock $ENV{SIZE_FLAG} ${CMAKE_SOURCE_DIR}/data/a)
add_test(NAME b COMMAND $<TARGET_FILE:cdmf> -k 10 -t 5 -T 3 -l 0.05 -r 1 -q 1 -P 1 -V $ENV{VERSION_FLAG} -nThreadsPerBlock $ENV{SIZE_FLAG} ${CMAKE_SOURCE_DIR}/data/b)
add_test(NAME c COMMAND $<TARGET_FILE:cdmf> -k 10 -t 5 -T 3 -l 0.05 -r 1 -q 1 -P 1 -V $ENV{VERSION_FLAG} -nThreadsPerBlock $ENV{SIZE_FLAG} ${CMAKE_SOURCE_DIR}/data/c)
