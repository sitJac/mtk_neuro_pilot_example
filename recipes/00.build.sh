#!/bin/bash

# Remove and recreate build directory
rm -rf ../build && mkdir ../build

# Set Android NDK path if not already set
if [ -z "$ANDROID_NDK_HOME" ]; then
    export ANDROID_NDK_HOME="/home/xiejunjie/ToolChains/android-ndk-r25c"
fi
echo "Using NDK: $ANDROID_NDK_HOME"

# Choosing an ABI to build 
# armeabi-v7a for 32bit 
# arm64-v8a for 64bit
ANDROID_ABI=arm64-v8a

echo "Building ABI executable: $ANDROID_ABI"

# Configure with CMake
cmake -G "Unix Makefiles" -S ../src -B ../build \
 -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
 -DCMAKE_BUILD_TYPE=Debug \
 -DANDROID_ABI="$ANDROID_ABI" \
 -DANDROID_PLATFORM=android-29 \
 -DCMAKE_ANDROID_STL_TYPE=c++_static \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build the project
cmake --build ../build

# Check if build succeeded
if [ $? -ne 0 ]; then
    echo "Failed to build NeuroPilot Native Example"
    exit 1
fi

echo "Build succeeded!!! Push executables..."

# Push executables to Android device
adb push ../build/GenericClassifier /data/local/tmp/
adb shell chmod +x /data/local/tmp/GenericClassifier

adb push ../build/ClassifyImage /data/local/tmp/
adb shell chmod +x /data/local/tmp/ClassifyImage

adb push ../build/DetectObject /data/local/tmp/
adb shell chmod +x /data/local/tmp/DetectObject

echo "Done."