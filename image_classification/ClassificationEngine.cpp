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

#include <algorithm>

#include <android/NeuralNetworks.h>
#include <math.h>
#include <string.h>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>

#include "../logger/include/Logger.h"
#include "../profiler/include/Profiler.h"
#include "../utils/include/CommonDef.h"
#include "../utils/include/Utils.h"

#include "ClassificationEngine.h"

#define TAG "Engine"

namespace classification {

ClassificationEngine::ClassificationEngine(const std::string& model_path,
                                           const std::string& label_path, bool allow_fp16) {
    PROFILE_ME_AS("Construct engine");
    RETURN_IF_NP_ERROR(ANeuralNetworksTFLiteOptions_create(&mOptions));
    RETURN_IF_NP_ERROR(
        ANeuralNetworksTFLiteOptions_setAllowFp16PrecisionForFp32(mOptions, allow_fp16));
    RETURN_IF_NP_ERROR(
        ANeuralNetworksTFLiteOptions_setCacheDir(mOptions, "/data/local/tmp/"));
    // Default execution preference is Fast Single Answer. If user has power concern, user can choose Sustained Mode.
    RETURN_IF_NP_ERROR(
        ANeuralNetworksTFLiteOptions_setPreference(mOptions, ExecutionPreference::kSustainedSpeed));
    RETURN_IF_NP_ERROR(
        ANeuroPilotTFLiteWrapper_makeAdvTFLite(&mTFLite, model_path.c_str(), mOptions));
    PROFILE_END;
    ReadLabels(label_path);
    InitBuffers();
}

ClassificationEngine::ClassificationEngine(const char* model_buffer, size_t buffer_size,
                                           std::vector<std::string>& labels, bool allow_fp16) {
    PROFILE_ME_AS("Construct engine");
    RETURN_IF_NP_ERROR(ANeuralNetworksTFLiteOptions_create(&mOptions));
    RETURN_IF_NP_ERROR(
        ANeuralNetworksTFLiteOptions_setAllowFp16PrecisionForFp32(mOptions, allow_fp16));
    RETURN_IF_NP_ERROR(
        ANeuralNetworksTFLiteOptions_setCacheDir(mOptions, "/data/local/tmp/"));
    // Default execution preference is Fast Single Answer. If user has power concern, user can choose Sustained Mode.
    RETURN_IF_NP_ERROR(
        ANeuralNetworksTFLiteOptions_setPreference(mOptions, ExecutionPreference::kSustainedSpeed));
    RETURN_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_makeAdvTFLiteWithBuffer(&mTFLite, model_buffer,
                                                                        buffer_size, mOptions));
    PROFILE_END;
    mLabels.assign(labels.begin(), labels.end());
    InitBuffers();
}

ClassificationEngine::~ClassificationEngine() {
    if (mTFLite != nullptr) {
        ANeuroPilotTFLiteWrapper_free(mTFLite);
        mTFLite = nullptr;
    }
    if (mOptions != nullptr) {
        ANeuralNetworksTFLiteOptions_free(mOptions);
        mOptions = nullptr;
    }
    if (mOutputBuffer != nullptr) {
        free(mOutputBuffer);
    }
}

void ClassificationEngine::InitBuffers() {
    // Get the input tensor size
    RETURN_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorByteSize(mTFLite, TFLITE_BUFFER_TYPE_INPUT,
                                                                  0, &mInputTensorByteSize));

    // Get the output tensor size and allocate buffer
    RETURN_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorByteSize(
        mTFLite, TFLITE_BUFFER_TYPE_OUTPUT, 0, &mOutputTensorByteSize));

    mOutputBuffer = reinterpret_cast<void*>(calloc(1, mOutputTensorByteSize));
    if (mOutputBuffer == nullptr) {
        LOG_ERROR(TAG, "Fail to allocate memory for post-processing");
    }
}

bool ClassificationEngine::Inference(const void* buffer, const std::size_t buffer_size,
                                     std::vector<std::string>& results) {
    std::size_t in_tensor_size = 0;
    TFLiteTensorType tensor_type = TFLITE_TENSOR_TYPE_NONE;
    std::vector<std::pair<float, int>> top_results;

    // Check if the input buffer size is matched with the input tensor size
    if (buffer_size != mInputTensorByteSize) {
        LOG_ERROR(TAG, "Input buffer size(%d) != Input tensor size(%d)", buffer_size,
                  mInputTensorByteSize);
        return false;
    }

    if (mOutputBuffer == nullptr) {
        LOG_ERROR(TAG, "NULL output buffer");
        return false;
    }

    // Copy input buffer to input tensor
    {
        PROFILE_ME_AS("Copy input");
        RETURN_FALSE_IF_NP_ERROR(
            ANeuroPilotTFLiteWrapper_setInputTensorData(mTFLite, 0, buffer, buffer_size));
    }

    // Start inference
    for (auto i = 0; i < mLoopCount; i++) {
        PROFILE_ME_AS("Invoke");
        RETURN_FALSE_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_invoke(mTFLite));
    }

    // Copy the output tensor data to allocated buffer
    {
        PROFILE_ME_AS("Copy output");
        RETURN_FALSE_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getOutputTensorData(
            mTFLite, 0, mOutputBuffer, mOutputTensorByteSize));
    }

    // Get data type of the output tensor
    RETURN_FALSE_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorType(
        mTFLite, TFLITE_BUFFER_TYPE_OUTPUT, 0, &tensor_type));
    if (tensor_type == TFLITE_TENSOR_TYPE_FLOAT) {
        GetTopN<float>(reinterpret_cast<float*>(mOutputBuffer),
                       mOutputTensorByteSize / sizeof(float), mNumberOfResults, mThreshold,
                       &top_results, true);
    } else if (tensor_type == TFLITE_TENSOR_TYPE_UINT8) {
        GetTopN<uint8_t>(reinterpret_cast<uint8_t*>(mOutputBuffer), mOutputTensorByteSize,
                         mNumberOfResults, mThreshold, &top_results, false);
    } else {
        LOG_ERROR(TAG, "Unsupported output tensor data type");
        return false;
    }

    for (const auto& result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        LOG_VERBOSE_IF(mVerboseLog, TAG, "Confidence: %f, index:%d, label:%s", confidence, index,
                       mLabels[index].c_str());
        results.emplace_back(mLabels[index]);
    }
    return true;
}

ModelDataType ClassificationEngine::GetModelInputDataType() {
    TFLiteTensorType tensor_type = TFLITE_TENSOR_TYPE_NONE;

    if (ANeuroPilotTFLiteWrapper_getTensorType(mTFLite, TFLITE_BUFFER_TYPE_INPUT, 0,
                                               &tensor_type) == ANEURALNETWORKS_NO_ERROR) {
        if (tensor_type == TFLITE_TENSOR_TYPE_FLOAT) {
            return ModelDataType::TYPE_FLOAT_32;
        } else if (tensor_type == TFLITE_TENSOR_TYPE_UINT8) {
            return ModelDataType::TYPE_UINT8;
        }
    }
    return ModelDataType::TYPE_UNKNOWN;
}

ModelDataType ClassificationEngine::GetModelOutputDataType() {
    TFLiteTensorType tensor_type = TFLITE_TENSOR_TYPE_NONE;

    if (ANeuroPilotTFLiteWrapper_getTensorType(mTFLite, TFLITE_BUFFER_TYPE_OUTPUT, 0,
                                               &tensor_type) == ANEURALNETWORKS_NO_ERROR) {
        if (tensor_type == TFLITE_TENSOR_TYPE_FLOAT) {
            return ModelDataType::TYPE_FLOAT_32;
        } else if (tensor_type == TFLITE_TENSOR_TYPE_UINT8) {
            return ModelDataType::TYPE_UINT8;
        }
    }
    return ModelDataType::TYPE_UNKNOWN;
}

bool ClassificationEngine::ReadLabels(const std::string& label_path) {
    return utils::ReadFileByLine(label_path, &mLabels);
}

/* Returns the top N confidence values over threshold in the provided vector,
 * sorted by confidence in descending order.
 */
template <class T>
void ClassificationEngine::GetTopN(T* prediction, int prediction_size, size_t num_results,
                                   float threshold, std::vector<std::pair<float, int>>* top_results,
                                   bool input_floating) {
    // Will contain top N results in ascending order.
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                        std::greater<std::pair<float, int>>>
        top_result_pq;
    for (int i = 0; i < prediction_size; ++i) {
        float value;
        if (input_floating) {
            value = prediction[i];
        } else {
            value = prediction[i] / 255.0;
        }

        // Only add it if it beats the threshold and has a chance at being in
        // the top N.
        if (value < threshold) {
            continue;
        }
        top_result_pq.push(std::pair<float, int>(value, i));

        // If at capacity, kick the smallest value out.
        if (top_result_pq.size() > num_results) {
            top_result_pq.pop();
        }
    }

    // Copy to output vector and reverse into descending order.
    while (!top_result_pq.empty()) {
        top_results->push_back(top_result_pq.top());
        top_result_pq.pop();
    }
    std::reverse(top_results->begin(), top_results->end());
}

}  // namespace classification
