# Build and Run Recipes

This folder contains all build and execution scripts for the NeuroPilot Android Native Sample.

## Scripts

### Build Scripts
- `00.build.bat` - Windows batch script for building the project
- `00.build.sh` - Linux shell script for building the project

### Execution Scripts
- `01.run_basic_example.bat/.sh` - Run basic MNIST classification example
- `02.run_classify_example.bat/.sh` - Run MobileNet image classification example  
- `03.run_detection_example.bat/.sh` - Run SSD object detection example

## Usage

### On Linux/Ubuntu:
```bash
cd recipes
./00.build.sh              # Build the project
./01.run_basic_example.sh   # Run basic example
./02.run_classify_example.sh # Run classification example
./03.run_detection_example.sh # Run detection example
```

### On Windows:
```cmd
cd recipes
00.build.bat              # Build the project
01.run_basic_example.bat   # Run basic example
02.run_classify_example.bat # Run classification example
03.run_detection_example.bat # Run detection example
```

## Prerequisites

1. Android NDK installed and `ANDROID_NDK_HOME` environment variable set
2. CMake installed
3. Android device connected via ADB with USB debugging enabled
4. Device should be rooted for the examples to work properly

## Notes

All paths in the scripts are relative to the parent directory (one level up from recipes folder), so the scripts must be run from within the recipes directory.