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
#include "DetectionEngine.h"

#define TAG "Engine"

namespace detection {

// SSD Mobilenet V1 Model assumes class 0 is background class
// in label file and class labels start from 1 to number_of_classes+1,
// while output classes correspond to class index from 0 to number_of_classes
constexpr int kLabelOffset = 1;

DetectionEngine::DetectionEngine(const std::string& model_path, const std::string& label_path,
                                 bool allow_fp16) {
    PROFILE_ME_AS("Construct engine");
    RETURN_IF_NP_ERROR(ANeuralNetworksTFLiteOptions_create(&mOptions));
    RETURN_IF_NP_ERROR(
        ANeuralNetworksTFLiteOptions_setCacheDir(mOptions, "/data/local/tmp/"));
    // Default execution preference is Fast Single Answer. If user has power concern, user can choose Sustained Mode.
    RETURN_IF_NP_ERROR(
        ANeuralNetworksTFLiteOptions_setPreference(mOptions, ExecutionPreference::kSustainedSpeed));
    RETURN_IF_NP_ERROR(
        ANeuralNetworksTFLiteOptions_setAllowFp16PrecisionForFp32(mOptions, allow_fp16));
    RETURN_IF_NP_ERROR(
        ANeuroPilotTFLiteWrapper_makeAdvTFLite(&mTFLite, model_path.c_str(), mOptions));
    PROFILE_END;
    ReadLabels(label_path);
    InitBuffers();
}

DetectionEngine::DetectionEngine(const char* model_buffer, size_t buffer_size,
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

DetectionEngine::~DetectionEngine() {
    if (mTFLite != nullptr) {
        ANeuroPilotTFLiteWrapper_free(mTFLite);
        mTFLite = nullptr;
    }
    if (mOptions != nullptr) {
        ANeuralNetworksTFLiteOptions_free(mOptions);
        mOptions = nullptr;
    }
    if (mOutputBoxesBuffer != nullptr) {
        free(mOutputBoxesBuffer);
        mOutputBoxesBuffer  =nullptr;
    }
    if (mOutputClassesBuffer != nullptr) {
        free(mOutputClassesBuffer);
        mOutputClassesBuffer = nullptr;
    }
    if (mOutputScoresBuffer != nullptr) {
        free(mOutputScoresBuffer);
        mOutputScoresBuffer = nullptr;
    }
    if (mOutputDetectionsBuffer != nullptr) {
        free(mOutputDetectionsBuffer);
        mOutputDetectionsBuffer = nullptr;
    }
}

void DetectionEngine::InitBuffers() {
    int rank = 0;
    int dims[4];
    // Get the input tensor size
    RETURN_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorByteSize(mTFLite, TFLITE_BUFFER_TYPE_INPUT,
                                                                  0, &mInputTensorByteSize));

    // Get the output tensor size and allocate buffer
    RETURN_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorByteSize(
        mTFLite, TFLITE_BUFFER_TYPE_OUTPUT, OUTPUT_TENSOR_DETECT_BOXES, &mOutputBoxesByteSize));

    mOutputBoxesBuffer = reinterpret_cast<void*>(calloc(1, mOutputBoxesByteSize));
    if (mOutputBoxesBuffer == nullptr) {
        LOG_ERROR(TAG, "Fail to allocate memory for output boxes");
    }

    RETURN_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorByteSize(
        mTFLite, TFLITE_BUFFER_TYPE_OUTPUT, OUTPUT_TENSOR_DETECT_CLASSES, &mOutputClassesByteSize));

    mOutputClassesBuffer = reinterpret_cast<void*>(calloc(1, mOutputClassesByteSize));
    if (mOutputClassesBuffer == nullptr) {
        LOG_ERROR(TAG, "Fail to allocate memory for output classes");
    }

    RETURN_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorByteSize(
        mTFLite, TFLITE_BUFFER_TYPE_OUTPUT, OUTPUT_TENSOR_DETECT_SCORES, &mOutputScoresByteSize));

    mOutputScoresBuffer = reinterpret_cast<void*>(calloc(1, mOutputScoresByteSize));
    if (mOutputScoresBuffer == nullptr) {
        LOG_ERROR(TAG, "Fail to allocate memory for output scores");
    }

    RETURN_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorByteSize(
        mTFLite, TFLITE_BUFFER_TYPE_OUTPUT, OUTPUT_TENSOR_NUM_DETECTIONS,
        &mOutputDetectionsByteSize));

    mOutputDetectionsBuffer = reinterpret_cast<void*>(calloc(1, mOutputDetectionsByteSize));
    if (mOutputDetectionsBuffer == nullptr) {
        LOG_ERROR(TAG, "Fail to allocate memory for output detections");
    }

    RETURN_IF_NP_ERROR(
        ANeuroPilotTFLiteWrapper_getTensorRank(mTFLite, TFLITE_BUFFER_TYPE_INPUT, 0, &rank));
    if (rank != 4) {
        LOG_ERROR(TAG, "The number of rank of input tensor should be 4");
    }
    RETURN_IF_NP_ERROR(
        ANeuroPilotTFLiteWrapper_getTensorDimensions(mTFLite, TFLITE_BUFFER_TYPE_INPUT, 0, dims));
    mInputHeight = dims[1];
}

bool DetectionEngine::Inference(const void* buffer, const std::size_t buffer_size,
                                std::vector<recognition::Recognition>& results) {
    std::size_t in_tensor_size = 0;
    TFLiteTensorType tensor_type = TFLITE_TENSOR_TYPE_NONE;
    std::vector<std::pair<float, int>> top_results;

    // Check if the input buffer size is matched with the input tensor size
    if (buffer_size != mInputTensorByteSize) {
        LOG_ERROR(TAG, "Input buffer size(%d) != Input tensor size(%d)", buffer_size,
                  mInputTensorByteSize);
        return false;
    }

    if (mOutputBoxesBuffer == nullptr) {
        LOG_ERROR(TAG, "NULL output boxes buffer");
        return false;
    }

    if (mOutputClassesBuffer == nullptr) {
        LOG_ERROR(TAG, "NULL output classes buffer");
        return false;
    }

    if (mOutputScoresBuffer == nullptr) {
        LOG_ERROR(TAG, "NULL output scores buffer");
        return false;
    }

    if (mOutputDetectionsBuffer == nullptr) {
        LOG_ERROR(TAG, "NULL output detections buffer");
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
        PROFILE_ME_AS("Copy output boxes");
        RETURN_FALSE_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getOutputTensorData(
            mTFLite, OUTPUT_TENSOR_DETECT_BOXES, mOutputBoxesBuffer, mOutputBoxesByteSize));
    }

    {
        PROFILE_ME_AS("Copy output classes");
        RETURN_FALSE_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getOutputTensorData(
            mTFLite, OUTPUT_TENSOR_DETECT_CLASSES, mOutputClassesBuffer, mOutputClassesByteSize));
    }

    {
        PROFILE_ME_AS("Copy output scores");
        RETURN_FALSE_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getOutputTensorData(
            mTFLite, OUTPUT_TENSOR_DETECT_SCORES, mOutputScoresBuffer, mOutputScoresByteSize));
    }

    {
        PROFILE_ME_AS("Copy output detections");
        RETURN_FALSE_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getOutputTensorData(
            mTFLite, OUTPUT_TENSOR_NUM_DETECTIONS, mOutputDetectionsBuffer,
            mOutputDetectionsByteSize));
    }

    GetRecognitions<float>(reinterpret_cast<float*>(mOutputBoxesBuffer),
                           reinterpret_cast<float*>(mOutputClassesBuffer),
                           reinterpret_cast<float*>(mOutputScoresBuffer),
                           reinterpret_cast<float*>(mOutputDetectionsBuffer), &results);

    return true;
}

ModelDataType DetectionEngine::GetModelInputDataType() {
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

ModelDataType DetectionEngine::GetModelOutputDataType(int index) {
    TFLiteTensorType tensor_type = TFLITE_TENSOR_TYPE_NONE;

    if (ANeuroPilotTFLiteWrapper_getTensorType(mTFLite, TFLITE_BUFFER_TYPE_OUTPUT, index,
                                               &tensor_type) == ANEURALNETWORKS_NO_ERROR) {
        if (tensor_type == TFLITE_TENSOR_TYPE_FLOAT) {
            return ModelDataType::TYPE_FLOAT_32;
        } else if (tensor_type == TFLITE_TENSOR_TYPE_UINT8) {
            return ModelDataType::TYPE_UINT8;
        }
    }
    return ModelDataType::TYPE_UNKNOWN;
}

bool DetectionEngine::ReadLabels(const std::string& label_path) {
    return utils::ReadFileByLine(label_path, &mLabels);
}

template <class T>
void DetectionEngine::GetRecognitions(T* boxes, T* classes, T* scores, T* detections,
                                      std::vector<recognition::Recognition>* recognitions) {
    if (!std::is_same<T, float>::value) {
        LOG_ERROR(TAG, "Only support float values");
        return;
    }

    for (int i = 0; i < DETECT_NUMBER_OF_DETECTIONS; i++) {
        recognition::Rect rect = {
            boxes[1 + (i * 4)] * mInputHeight, boxes[0 + (i * 4)] * mInputHeight,
            boxes[3 + (i * 4)] * mInputHeight, boxes[2 + (i * 4)] * mInputHeight};
        recognition::Recognition recognition(classes[i] + kLabelOffset,
                                             mLabels[static_cast<int>(classes[i]) + kLabelOffset],
                                             scores[i], rect);

        LOG_VERBOSE_IF(mVerboseLog, TAG,
                       "Confidence: %f, index:%d, label:%s, left:%f, top:%f, right:%f, bottom:%f",
                       recognition.GetConfidence(), recognition.GetClass(),
                       recognition.GetTitle().c_str(), recognition.GetRect().left,
                       recognition.GetRect().top, recognition.GetRect().right,
                       recognition.GetRect().bottom);
        // Add recognition which is above the threshold
        if (scores[i] >= mThreshold) {
            recognitions->emplace_back(recognition);
        }
    }
}

}  // namespace detection
