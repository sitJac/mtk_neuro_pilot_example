rmdir /s /q ..\build && mkdir ..\build

@echo off
IF NOT DEFINED ANDROID_NDK_HOME (
    set ANDROID_NDK_HOME=/home/xiejunjie/ToolChains/android-ndk-r25c
)
echo Using NDK: %ANDROID_NDK_HOME%

@REM Choosing an ABI to build 
@REM armeabi-v7a for 32bit 
@REM arm64-v8a for 64bit
set ANDROID_ABI=arm64-v8a

echo Building ABI executable: %ANDROID_ABI%

cmake -G"MinGW Makefiles" -S ..\src -B ..\build ^
 -DCMAKE_TOOLCHAIN_FILE=%ANDROID_NDK_HOME%/build/cmake/android.toolchain.cmake ^
 -DCMAKE_BUILD_TYPE=Debug ^
 -DANDROID_ABI=%ANDROID_ABI% ^
 -DANDROID_PLATFORM=android-29 ^
 -DCMAKE_ANDROID_STL_TYPE=c++_static ^
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build ..\build

if %ERRORLEVEL% neq 0 (
  echo "Failed to build NeuroPilot Native Example"
  exit /b %ERRORLEVEL%
)

echo Build succeeded!!! Push executables...

adb push ..\build\GenericClassifier /data/local/tmp/
adb shell chmod +x /data/local/tmp/GenericClassifier

adb push ..\build\ClassifyImage /data/local/tmp/
adb shell chmod +x /data/local/tmp/ClassifyImage

adb push ..\build\DetectObject /data/local/tmp/
adb shell chmod +x /data/local/tmp/DetectObject

echo Done.

