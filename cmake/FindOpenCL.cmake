################################################################################
# Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

function(_FIND_OPENCL_VERSION)
	include(CheckSymbolExists)
	include(CMakePushCheckState)
	set(CMAKE_REQUIRED_QUIET ${OpenCL_FIND_QUIETLY})

	CMAKE_PUSH_CHECK_STATE()
	foreach(VERSION "2_2" "2_1" "2_0" "1_2" "1_1" "1_0")
		set(CMAKE_REQUIRED_INCLUDES "${OPENCL_INCLUDE_DIR}")

		if(APPLE)
			CHECK_SYMBOL_EXISTS(
					CL_VERSION_${VERSION}
					"${OPENCL_INCLUDE_DIR}/Headers/cl.h"
					OPENCL_VERSION_${VERSION})
		else()
			CHECK_SYMBOL_EXISTS(
					CL_VERSION_${VERSION}
					"${OPENCL_INCLUDE_DIR}/CL/cl.h"
					OPENCL_VERSION_${VERSION})
		endif()

		if(OPENCL_VERSION_${VERSION})
			string(REPLACE "_" "." VERSION "${VERSION}")
			set(OpenCL_VERSION_STRING ${VERSION} PARENT_SCOPE)
			string(REGEX MATCHALL "[0-9]+" version_components "${VERSION}")
			list(GET version_components 0 major_version)
			list(GET version_components 1 minor_version)
			set(OpenCL_VERSION_MAJOR ${major_version} PARENT_SCOPE)
			set(OpenCL_VERSION_MINOR ${minor_version} PARENT_SCOPE)
			break()
		endif()
	endforeach()
	CMAKE_POP_CHECK_STATE()
endfunction()

find_path(OPENCL_INCLUDE_DIR
	NAMES OpenCL/cl.h CL/cl.h
	HINTS
	${OPENCL_ROOT}/include
	$ENV{AMDAPPSDKROOT}/include
	$ENV{CUDA_PATH}/include
	PATHS
	/usr/include
	/usr/local/include
	/usr/local/cuda/include
	/opt/cuda/include
	/opt/rocm/opencl/include
	DOC "OpenCL header file path"
	)
mark_as_advanced( OPENCL_INCLUDE_DIR )

_FIND_OPENCL_VERSION()

if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
	find_library( OPENCL_LIBRARIES
		NAMES OpenCL
		HINTS
		${OPENCL_ROOT}/lib
		$ENV{AMDAPPSDKROOT}/lib
		$ENV{CUDA_PATH}/lib
		DOC "OpenCL dynamic library path"
		PATH_SUFFIXES x86_64 x64 x86_64/sdk x86_64-linux-gnu
		PATHS
		/opt/intel/opencl-1.2-6.4.0.37/lib64
		/usr/lib
		/usr/local/cuda/lib
		/opt/cuda/lib
		/opt/rocm/opencl/lib
		)
else( )
	find_library( OPENCL_LIBRARIES
		NAMES OpenCL
		HINTS
		${OPENCL_ROOT}/lib
		$ENV{AMDAPPSDKROOT}/lib
		$ENV{CUDA_PATH}/lib
		DOC "OpenCL dynamic library path"
		PATH_SUFFIXES x86 Win32
		PATHS
		/opt/intel/opencl-1.2-6.4.0.37/lib64
		/usr/lib
		/usr/local/cuda/lib
		/opt/cuda/lib
		)
endif( )
mark_as_advanced( OPENCL_LIBRARIES )

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( OPENCL DEFAULT_MSG OPENCL_LIBRARIES OPENCL_INCLUDE_DIR )

set(OpenCL_FOUND ${OPENCL_FOUND} CACHE INTERNAL "")
set(OpenCL_LIBRARIES ${OPENCL_LIBRARIES} CACHE INTERNAL "")
set(OpenCL_INCLUDE_DIR ${OPENCL_INCLUDE_DIR} CACHE INTERNAL "")

if( NOT OPENCL_FOUND )
	message( STATUS "FindOpenCL looked for libraries named: OpenCL" )
endif()
