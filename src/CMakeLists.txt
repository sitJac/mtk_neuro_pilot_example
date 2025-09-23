cmake_minimum_required(VERSION 3.6.0)

project("neuropilot")

add_library(utils STATIC
    utils/Utils.cpp
)

add_library(logger STATIC
    logger/Logger.cpp
    logger/LoggerImpl.cpp
)

add_library(profiler STATIC
    profiler/Stopwatch.cpp
    profiler/ScopeProfiler.cpp
)

add_executable(GenericClassifier
    basic/Main.cpp
    basic/GenericClassifier.cpp
)

add_executable(ClassifyImage
    image_classification/Main.cpp
    image_classification/ClassificationEngine.cpp
)

add_executable(DetectObject
    object_detection/Main.cpp
    object_detection/DetectionEngine.cpp
)

target_link_libraries(GenericClassifier PRIVATE
    logger log
)

target_link_libraries(ClassifyImage PRIVATE
    utils logger profiler log
)

target_link_libraries(DetectObject PRIVATE
    utils logger profiler log
)
