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

#define PREDICT_INPUT_MEAN (127.5f)
#define PREDICT_INPUT_STD (127.5f)
#define PREDICT_THRESHOLD (0.001f)
#define PREDICT_NUMBER_OF_RESULTS 5

namespace classification {

enum class ModelDataType {
    TYPE_UNKNOWN = 0,
    TYPE_FLOAT_32,
    TYPE_UINT8,
};

class ClassificationEngine {
public:
    explicit ClassificationEngine(const std::string& model_path, const std::string& label_path,
                                  bool allow_fp16 = true);

    ClassificationEngine(const char* model_buffer, size_t buffer_size,
                         std::vector<std::string>& labels, bool allow_fp16 = true);

    bool Inference(const void* buffer, const std::size_t buffer_size,
                   std::vector<std::string>& results);

    ~ClassificationEngine();

    uint8_t GetNumberOfResults() { return mNumberOfResults; }

    void SetNumberOfResults(size_t num) { mNumberOfResults = num; }

    ModelDataType GetModelInputDataType();

    ModelDataType GetModelOutputDataType();

    void SetVerboseLog(bool enable) { mVerboseLog = enable; }

    void SetLoopCount(int count) { mLoopCount = count; }

private:
    bool ReadLabels(const std::string& label_path);

    void InitBuffers();

    template <class T>
    void GetTopN(T* prediction, int prediction_size, size_t num_results, float threshold,
                 std::vector<std::pair<float, int>>* top_results, bool input_floating);

private:
    ANeuralNetworksTFLite* mTFLite = nullptr;
    ANeuralNetworksTFLiteOptions* mOptions = nullptr;
    void* mOutputBuffer = nullptr;
    size_t mInputTensorByteSize = 0;
    size_t mOutputTensorByteSize = 0;
    std::vector<std::string> mLabels;
    float mThreshold = PREDICT_THRESHOLD;
    size_t mNumberOfResults = PREDICT_NUMBER_OF_RESULTS;
    bool mVerboseLog = false;
    int mLoopCount = 1;
};

}  // namespace classification
