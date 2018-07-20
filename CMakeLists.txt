# Minimal version of CMake
cmake_minimum_required (VERSION 3.9)

if(UNIX AND NOT APPLE)
    message("Importing alternative FindOpenCL for Linux")
    set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
endif(UNIX AND NOT APPLE)

# Define project name
project(cdmf)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 14)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/exec/")

#set(CMAKE_VERBOSE_MAKEFILE ON)

set(NUMRUN "-DNUM_RUN=1")
#set(WARNINGS "-Wno-format -Wno-unused-result -Wno-write-strings -Wno-deprecated-declarations")
#set(WARNINGS "-Wno-format -Wno-unused-result -Wno-write-strings")
set(VALUE "-DVALUE_TYPE=double")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${NUMRUN} ${VERSION} ${VALUE} -m64 ${WARNINGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${NUMRUN} ${VERSION} ${VALUE} -m64 ${WARNINGS}")


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
include_directories(${OpenCL_INCLUDE_DIR})
add_executable(cdmf ${source_src} ${source_hdr})
target_link_libraries(cdmf ${OpenCL_LIBRARIES} OpenMP::OpenMP_CXX)

enable_testing()
add_test(a ../exec/cdmf -k 40 -t 5 -T 5 -n 24 -l 0.05 ../data/a)
add_test(b ../exec/cdmf -k 40 -t 5 -T 5 -n 24 -l 0.05 ../data/b)
add_test(c ../exec/cdmf -k 40 -t 5 -T 5 -n 24 -l 0.05 ../data/c)
