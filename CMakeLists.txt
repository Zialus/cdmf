# Minimal version of CMake
cmake_minimum_required(VERSION 3.9)

# Define project name
project(cdmf)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 14)

#set(CMAKE_VERBOSE_MAKEFILE ON)

# Add some compilation flags
set(VALUE "-DVALUE_TYPE=float")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${VALUE}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VALUE}")

# Compiler flags to enable more warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif ()

# Compiler flags to enable all warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall")
endif ()

# Compiler flags to silence/or add some warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44365")
endif ()

# Silence warning C4996 on Microsoft Compilers
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif ()

# Source code
file(GLOB source_src "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB source_hdr "${PROJECT_SOURCE_DIR}/src/*.h" "${PROJECT_SOURCE_DIR}/src/detail/*.h" "${PROJECT_SOURCE_DIR}/src/detail/opencl/*.h")

# Executable and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/exec/")
add_executable(cdmf ${source_src} ${source_hdr})
target_link_libraries(cdmf OpenCL::OpenCL OpenMP::OpenMP_CXX)

# Tests
enable_testing()
add_test(NAME a COMMAND $<TARGET_FILE:cdmf> -k 10 -t 5 -T 3 -l 0.05 -p 1 -r 1 -q 1 -P 0 -d 1 -V $ENV{VERSION_FLAG} -nThreadsPerBlock $ENV{SIZE_FLAG} ${CMAKE_SOURCE_DIR}/data/a)
add_test(NAME b COMMAND $<TARGET_FILE:cdmf> -k 10 -t 5 -T 3 -l 0.05 -p 1 -r 1 -q 1 -P 0 -d 1 -V $ENV{VERSION_FLAG} -nThreadsPerBlock $ENV{SIZE_FLAG} ${CMAKE_SOURCE_DIR}/data/b)
add_test(NAME c COMMAND $<TARGET_FILE:cdmf> -k 10 -t 5 -T 3 -l 0.05 -p 1 -r 1 -q 1 -P 0 -d 1 -V $ENV{VERSION_FLAG} -nThreadsPerBlock $ENV{SIZE_FLAG} ${CMAKE_SOURCE_DIR}/data/c)
add_test(NAME simple COMMAND $<TARGET_FILE:cdmf> -k 10 -t 5 -T 3 -l 0.05 -p 1 -r 1 -q 1 -P 0 -d 1 -V $ENV{VERSION_FLAG} -nThreadsPerBlock $ENV{SIZE_FLAG} ${CMAKE_SOURCE_DIR}/data/simple)
add_test(NAME tricky_simple COMMAND $<TARGET_FILE:cdmf> -k 10 -t 5 -T 3 -l 0.05 -p 1 -r 1 -q 1 -P 0 -d 1 -V $ENV{VERSION_FLAG} -nThreadsPerBlock $ENV{SIZE_FLAG} ${CMAKE_SOURCE_DIR}/data/tricky_simple)
add_test(NAME toy_example COMMAND $<TARGET_FILE:cdmf> -k 10 -t 5 -T 3 -l 0.05 -p 1 -r 1 -q 1 -P 0 -d 1 -V $ENV{VERSION_FLAG} -nThreadsPerBlock $ENV{SIZE_FLAG} ${CMAKE_SOURCE_DIR}/data/toy_example)
