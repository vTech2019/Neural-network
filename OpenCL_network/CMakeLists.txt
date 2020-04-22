
cmake_minimum_required (VERSION 3.1)
project (OpenCL C)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../libs/ppm_reader" build)

set(-DCMAKE_GENERATOR_PLATFORM x64)

if (linux)
    set(OpenCL_INCLUDE_DIR "/usr/local/cuda/include")
    set(OpenCL_LIBRARY "/usr/local/cuda/lib64/libOpenCL.so")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -Wall -Wextra -fexceptions -lm -O0  -march=native -ftree-vectorize --param l2-cache-size=25384")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package(OpenCL)
if (OpenCL_FOUND)
    message(STATUS "OpenCL found: ${OPENCL_FOUND}")
    message(STATUS "OpenCL includes: ${OPENCL_INCLUDE_DIRS}")
    message(STATUS "OpenCL libraries: ${OPENCL_LIBRARIES}")
endif()

file(GLOB_RECURSE  SOURCES
${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
)


add_executable (${PROJECT_NAME}  ${SOURCES})
message("CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}" )

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCL_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} ppm_reader ${OpenCL_LIBRARY})
if (UNIX)
    target_link_libraries(${PROJECT_NAME} m)
endif()