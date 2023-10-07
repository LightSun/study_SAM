cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)

project(SamCPP)

#find_package(OpenCV CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)

if($ENV{OpenCV_DIR})
    find_package(OpenCV 4 REQUIRED)
else()
    #SET(OPENCV /home/heaven7/heaven7/libs/opencv-3.4.7/opencv-3.4.7/_build)
    SET(OPENCV /home/heaven7/heaven7/libs/opencv-3.4.7/opencv-4.7.0/_install/lib64/cmake/opencv4)
    include(${OPENCV}/OpenCVConfig.cmake)
endif()
include_directories( ${OpenCV_INCLUDE_DIRS} )

##
add_library(sam_cpp_lib SHARED sam.h sam.cpp)
set(ONNXRUNTIME_ROOT_DIR "/home/heaven7/heaven7/work/TensorRT/onnxruntime-linux-x64-gpu-1.15.1")
if (WIN32)
  set(onnxruntime_lib ${VCPKG_INSTALLED_DIR}/x64-windows/lib/onnxruntime.lib)
else()
  set(onnxruntime_lib ${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so)
  target_include_directories(sam_cpp_lib PRIVATE ${ONNXRUNTIME_ROOT_DIR}/include)
endif()
target_link_libraries(sam_cpp_lib PRIVATE
  ${onnxruntime_lib}
  ${OpenCV_LIBS}
)

add_executable(sam_cpp_test test.cpp)
target_link_libraries(sam_cpp_test PRIVATE
  sam_cpp_lib
  ${OpenCV_LIBS}
  gflags
)
