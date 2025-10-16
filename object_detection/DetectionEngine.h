/*
 * Copyright (C) 2019 MediaTek Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License
 *
 * You may obtain a copy of the License at
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * Limitations under the License.
 */

#pragma once

#include <android/NeuralNetworks.h>
#include <cstdio>
#include <memory>
#include <vector>

#include "../neuropilot_api/NeuroPilotTFLiteShim.h"
#include "Recognition.h"

#define DETECT_INPUT_MEAN (127.5f)
#define DETECT_INPUT_STD (127.5f)
#define DETECT_THRESHOLD (0.5f)
#define DETECT_NUMBER_OF_DETECTIONS 10

#define OUTPUT_TENSOR_DETECT_BOXES 0
#define OUTPUT_TENSOR_DETECT_CLASSES 1
#define OUTPUT_TENSOR_DETECT_SCORES 2
#define OUTPUT_TENSOR_NUM_DETECTIONS 3

namespace detection {

enum class ModelDataType {
    TYPE_UNKNOWN = 0,
    TYPE_FLOAT_32,
    TYPE_UINT8,
};

class DetectionEngine {
public:
    explicit DetectionEngine(const std::string& model_path, const std::string& label_path,
                             bool allow_fp16 = true);

    DetectionEngine(const char* model_buffer, size_t buffer_size, std::vector<std::string>& labels,
                    bool allow_fp16 = true);

    bool Inference(const void* buffer, const std::size_t buffer_size,
                   std::vector<recognition::Recognition>& results);

    ~DetectionEngine();

    void SetThreshold(float threshold) { mThreshold = threshold; }

    float GetThreshold() const { return mThreshold; }

    ModelDataType GetModelInputDataType();

    ModelDataType GetModelOutputDataType(int index);

    void SetVerboseLog(bool enable) { mVerboseLog = enable; }

    void SetLoopCount(int count) { mLoopCount = count; }

private:
    bool ReadLabels(const std::string& label_path);

    void InitBuffers();

    template <class T>
    void GetRecognitions(T* boxes, T* classes, T* scores, T* detections,
                         std::vector<recognition::Recognition>* recognitions);

private:
    ANeuralNetworksTFLite* mTFLite = nullptr;
    ANeuralNetworksTFLiteOptions* mOptions = nullptr;
    std::vector<std::string> mLabels;
    std::vector<recognition::Recognition> mRecognitions;
    size_t mInputTensorByteSize = 0;
    void* mOutputBoxesBuffer = nullptr;
    size_t mOutputBoxesByteSize = 0;
    void* mOutputClassesBuffer = nullptr;
    size_t mOutputClassesByteSize = 0;
    void* mOutputScoresBuffer = nullptr;
    size_t mOutputScoresByteSize = 0;
    void* mOutputDetectionsBuffer = nullptr;
    size_t mOutputDetectionsByteSize = 0;
    float mThreshold = DETECT_THRESHOLD;
    size_t mNumberOfResults = DETECT_NUMBER_OF_DETECTIONS;
    bool mVerboseLog = false;
    int mLoopCount = 1;
    int mInputHeight = 0;
};

}  // namespace detection
