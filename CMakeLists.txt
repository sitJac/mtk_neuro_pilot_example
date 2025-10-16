cmake_minimum_required(VERSION 3.6.0)

project("neuropilot")

if(${TARGET} MATCHES "aarch64")
set(ANDROID_ABI arm64-v8a)
set(TARGET aarch64-linux-android29)
elseif(${TARGET} MATCHES "arm")
set(ANDROID_ABI armeabi-v7a)
set(TARGET armv7a-linux-androideabi29)
endif()

set(CMAKE_SYSTEM_NAME Android)
set(CMAKE_HOST_SYSTEM_NAME Linux)
set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION clang)
set(CMAKE_ANDROID_STL_TYPE c++_static)
set(ANDROID_PLATFORM android-29)
set(ANDROID_TOOLCHAIN clang)
set(CMAKE_BUILD_TYPE Release)
set(NDK_PATH /proj/mtk09942/android-ndk-r21d/)
include(${NDK_PATH}build/cmake/android.toolchain.cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__ANDROID_LINUX__ -Wno-attributes --target=${TARGET}")

add_library(
    utils
    STATIC
    utils/Utils.cpp)

add_library(
    logger
    STATIC
    logger/Logger.cpp
    logger/LoggerImpl.cpp)

add_library(
    profiler
    STATIC
    profiler/Stopwatch.cpp
    profiler/ScopeProfiler.cpp)

add_executable(
    GenericClassifier
    basic/Main.cpp
    basic/GenericClassifier.cpp)

add_executable(
    ClassifyImage
    image_classification/Main.cpp
    image_classification/ClassificationEngine.cpp)

add_executable(
    DetectObject
    object_detection/Main.cpp
    object_detection/DetectionEngine.cpp)

target_link_libraries(
    GenericClassifier
    logger
    log)

target_link_libraries(
    ClassifyImage
    utils
    logger
    profiler
    log)

target_link_libraries(
    DetectObject
    utils
    logger
    profiler
    log)
