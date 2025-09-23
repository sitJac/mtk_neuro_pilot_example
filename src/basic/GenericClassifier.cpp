/*
 * Copyright (C) 2020 MediaTek Inc.
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

#include <android/NeuralNetworks.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>

#include "../neuropilot_api/NeuroPilotTFLiteShim.h"
#include "../utils/include/CommonDef.h"

#include "GenericClassifier.h"

#define TAG "GenericClassifier"

namespace classifier {

GenericClassifier::GenericClassifier(const std::string& model_path) {
    ANeuralNetworksTFLiteOptions *tflite_options = nullptr;
    RETURN_IF_NP_ERROR(ANeuralNetworksTFLiteOptions_create(&tflite_options));
    RETURN_IF_NP_ERROR(ANeuralNetworksTFLiteOptions_setAccelerationMode(tflite_options, NP_ACCELERATION_NEURON));
    RETURN_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_makeAdvTFLite(&mTFLite, model_path.c_str(), tflite_options));
    RETURN_IF_NP_ERROR(ANeuralNetworksTFLiteOptions_free(tflite_options));

    // Get the input tensor size.
    RETURN_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorByteSize(mTFLite, TFLITE_BUFFER_TYPE_INPUT,
                                                                  0, &mInputTensorByteSize));

    // Get the output tensor size.
    RETURN_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorByteSize(
        mTFLite, TFLITE_BUFFER_TYPE_OUTPUT, 0, &mOutputTensorByteSize));

    // Get number of output classes.
    // For a classification model, the output dimension should be [1, Number of classes].
    // We can get the number of output classes from the output tensor byte size.
    // - uint8 model : Number of classes = output tensor byte size
    // - float model : Number of classes = output tensor byte size / sizeof(float)
    TFLiteTensorType tensor_type = TFLITE_TENSOR_TYPE_NONE;
    RETURN_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorType(mTFLite, TFLITE_BUFFER_TYPE_OUTPUT, 0,
                                                              &tensor_type));
    if (tensor_type == TFLITE_TENSOR_TYPE_UINT8) {
        mNumOfClasses = mOutputTensorByteSize;
    } else if (tensor_type == TFLITE_TENSOR_TYPE_FLOAT) {
        mNumOfClasses = mOutputTensorByteSize / sizeof(float);
    }

    LOG_INFO(TAG, "Input tensor byte size: %d", mInputTensorByteSize);
    LOG_INFO(TAG, "Output tensor byte size: %d", mOutputTensorByteSize);
}

GenericClassifier::~GenericClassifier() {
    if (mTFLite != nullptr) {
        ANeuroPilotTFLiteWrapper_free(mTFLite);
        mTFLite = nullptr;
    }
}

bool GenericClassifier::Inference(const std::string& input_path, const std::string& output_path,
                                  uint32_t* predicted_class) {
    bool ret = false;
    std::vector<uint8_t> input_buffer;
    std::vector<uint8_t> output_buffer;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::duration<double> duration_invoke;

    if (predicted_class == nullptr) {
        LOG_ERROR(TAG, "Null outpupt class");
        return ret;
    }

    if (!ReadInput(input_path, input_buffer)) {
        return ret;
    }
    // Copy the input buffer to the model input tensor.
    RETURN_FALSE_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_setInputTensorData(
        mTFLite, 0, reinterpret_cast<void*>(input_buffer.data()), mInputTensorByteSize));

    // Start the inference
    start = std::chrono::high_resolution_clock::now();
    RETURN_FALSE_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_invoke(mTFLite));
    duration_invoke = std::chrono::high_resolution_clock::now() - start;

    // Get the result from model output tensor.
    output_buffer.resize(mOutputTensorByteSize);
    RETURN_FALSE_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getOutputTensorData(
        mTFLite, 0, output_buffer.data(), mOutputTensorByteSize));

    // Save the result to the otuput file.
    SaveOutput(output_path, output_buffer);

    // Print the elasped time in ANeuroPilotTFLiteWrapper_invoke
    double duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(duration_invoke).count();
    LOG_INFO(TAG, "Duration in invoke: %.2f ms", duration);

    // Return the predicted class.
    return GetPredictedClass(predicted_class);
}

void GenericClassifier::PrintModelInformation() {
    /*
     * Get the following model information
     * - Data type
     * - Rank of input/output tensor
     * - Dimensions of input/output tensor
     */

    // Get the model data type
    TFLiteTensorType tensor_type = TFLITE_TENSOR_TYPE_NONE;

    RETURN_IF_NP_ERROR(
        ANeuroPilotTFLiteWrapper_getTensorType(mTFLite, TFLITE_BUFFER_TYPE_INPUT, 0, &tensor_type));
    if (tensor_type == TFLITE_TENSOR_TYPE_FLOAT) {
        LOG_INFO(TAG, "Input tensor data type: float");
    } else if (tensor_type == TFLITE_TENSOR_TYPE_UINT8) {
        LOG_INFO(TAG, "Input tensor data type: uint8");
    }
    RETURN_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorType(mTFLite, TFLITE_BUFFER_TYPE_OUTPUT, 0,
                                                              &tensor_type));
    if (tensor_type == TFLITE_TENSOR_TYPE_FLOAT) {
        LOG_INFO(TAG, "Output tensor data type: float");
    } else if (tensor_type == TFLITE_TENSOR_TYPE_UINT8) {
        LOG_INFO(TAG, "Output tensor data type: uint8");
    }

    // Get the rank of input tensor
    int input_rank = 0;
    RETURN_IF_NP_ERROR(
        ANeuroPilotTFLiteWrapper_getTensorRank(mTFLite, TFLITE_BUFFER_TYPE_INPUT, 0, &input_rank));
    // Get the dimensions of the input tensor
    std::vector<int> input_dims(input_rank, 0);
    RETURN_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorDimensions(
        mTFLite, TFLITE_BUFFER_TYPE_INPUT, 0, input_dims.data()));

    std::string input_dims_string = std::string("[");
    for (auto i = 0; i < input_rank; i++) {
        input_dims_string += std::to_string(input_dims[i]);
        if (i < input_rank - 1) {
            input_dims_string += std::string(" ,");
        }
    }
    input_dims_string += std::string("]");
    LOG_INFO(TAG, "Input tensor rank: %d, dimensions: %s", input_rank, input_dims_string.c_str());

    // Get the rank of output tensor
    int output_rank = 0;
    RETURN_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorRank(mTFLite, TFLITE_BUFFER_TYPE_OUTPUT, 0,
                                                              &output_rank));
    // Get the dimensions of the output tensor
    std::vector<int> output_dims(output_rank, 0);
    RETURN_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorDimensions(
        mTFLite, TFLITE_BUFFER_TYPE_OUTPUT, 0, output_dims.data()));

    std::string output_dims_string = std::string("[");
    for (auto i = 0; i < output_rank; i++) {
        output_dims_string += std::to_string(output_dims[i]);
        if (i < output_rank - 1) {
            output_dims_string += std::string(" ,");
        }
    }
    output_dims_string += std::string("]");
    LOG_INFO(TAG, "Output tensor rank: %d, dimensions: %s", output_rank,
             output_dims_string.c_str());
}

void GenericClassifier::EnableSoftmax(bool enable) {
    mApplySoftmax = enable;
}

bool GenericClassifier::ReadInput(const std::string& input_path, std::vector<uint8_t>& input) {
    std::size_t input_file_size = 0;
    std::ifstream file(input_path);
    if (!file) {
        LOG_ERROR(TAG, "Fail to read the input file.");
        return false;
    }

    // Clear the vector first
    input.clear();

    do {
        // Validate the input file size.
        file.seekg(0, std::ios::end);
        input_file_size = file.tellg();

        if (input_file_size != mInputTensorByteSize) {
            LOG_ERROR(
                TAG,
                "The size of the input file does not match the size of the model input tensor");
            break;
        }

        // Read the input file into a buffer.
        file.seekg(0, std::ios::beg);
        input.resize(input_file_size);
        file.read(reinterpret_cast<char*>(input.data()), input_file_size);
    } while (0);

    file.close();

    return (input.size() != 0);
}

bool GenericClassifier::SaveOutput(const std::string& output_path,
                                   const std::vector<uint8_t>& result) {
    if (result.size() == 0) {
        return false;
    }

    // Save the result to otuput file
    std::ofstream output_file(output_path, std::ios::out | std::ios::binary);
    if (!output_file) {
        LOG_ERROR(TAG, "Fail to save result to the output file.");
        return false;
    }
    output_file.write(reinterpret_cast<const char*>(result.data()), result.size());
    output_file.close();

    return true;
}

bool GenericClassifier::GetPredictedClass(uint32_t* predicted_class) {
    // Get the dequantized result from the model outupt tensor.
    std::vector<float> logits(mNumOfClasses, 0.0f);
    if (DequantizeOutput()) {
        RETURN_FALSE_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getDequantizedOutputByIndex(
            mTFLite, &logits[0], logits.size() * sizeof(float), 0));
    } else {
        RETURN_FALSE_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getOutputTensorData(
            mTFLite, 0, &logits[0], logits.size() * sizeof(float)));
    }

    typename std::vector<float>::iterator max_position =
        std::max_element(logits.begin(), logits.end());
    int argmax_id = std::distance(logits.begin(), max_position);

    int i = 0;
    if (mApplySoftmax) {
        // Take softmax by result.
        float max_value = logits[argmax_id];
        float exp_sum = 0.0f;

        for (auto& s : logits) {
            float f = exp(s - max_value);
            exp_sum += f;
        }
        // Print the probility of each class.
        for (auto& s : logits) {
            s /= exp_sum;
            LOG_INFO(TAG, "Probility[%d] = %.3f", i, s);
            i++;
        }
    } else {
        // Print the probility of each class.
        for (auto& s : logits) {
            LOG_INFO(TAG, "Probility[%d] = %.3f", i, s);
            i++;
        }
    }

    *predicted_class = static_cast<uint32_t>(argmax_id);
    return true;
}

bool GenericClassifier::DequantizeOutput() {
    bool do_dequantize = false;
    // Check output tensor data type
    TFLiteTensorType tensor_type = TFLITE_TENSOR_TYPE_NONE;
    RETURN_FALSE_IF_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorType(
        mTFLite, TFLITE_BUFFER_TYPE_OUTPUT, 0, &tensor_type));
    if (tensor_type == TFLITE_TENSOR_TYPE_UINT8) {
        do_dequantize = true;
    }

    return do_dequantize;
}

}  // namespace classifier
