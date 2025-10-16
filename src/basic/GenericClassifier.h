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

#pragma once

#include <string>
#include <vector>

#include "../neuropilot_api/NeuroPilotTFLiteShim.h"

namespace classifier {

class GenericClassifier {
   public:
    /**
     * @brief Construct a new General Classifier object
     *
     * @param model_path Path of the model file
     */
    explicit GenericClassifier(const std::string& model_path);

    /**
     * @brief Destroy the General Classifier object
     *
     */
    ~GenericClassifier();

    /**
     * @brief Inference with the given input file and save the result to the given output file.
     *
     * @param input_path Path of the input file.
     * @param output_path Path of the output file.
     * @param predicted_class The predicted class. The predicted class is meaningless if the
     * inference is failed.
     * @return true The inference is sueccessful.
     * @return false The inference is failed.
     */
    bool Inference(const std::string& input_path, const std::string& output_path,
                   uint32_t* predicted_class);

    /**
     * @brief Print logs to show the information of this model.
     *
     */
    void PrintModelInformation();

    /**
     * @brief Apply softmax on the inference result or not.
     *
     * @param enable True to apply softmax on the inference result.
     */
    void EnableSoftmax(bool enable);

   private:
    /**
     * @brief Read the raw data from the input file.
     *
     * @param input_path Path of the input file.
     * @param std::vector<uint8_t> Vector to load raw data from the input file.
     * @return true The raw data from the input file were successfully loaded into the vector.
     * @return false The raw data from the input file cannot be loaded into the vector.
     */
    bool ReadInput(const std::string& input_path, std::vector<uint8_t>& input);

    /**
     * @brief Save the inference result to the output file.
     *
     * @param output_path Path of the output file.
     * @param result The inference result from the model output tensor.
     * @return true The inference result were successfully saved to the output file.
     * @return false The inference result cannot be saved to the output file.
     */
    bool SaveOutput(const std::string& output_path, const std::vector<uint8_t>& result);

    /**
     * @brief Get the predicted class from the inference result.
     *
     * @return true Successfully obtained predicted class.
     * @return false The predicted class was not available.
     */
    bool GetPredictedClass(uint32_t* predicted_class);

    /**
     * @brief Check if we need to de-quantize the output tensor value to get a true floating point
     *        value.
     *
     * @return true Need to de-quantize the output tensor value.
     * @return false No need to de-quantize the output tensor value.
     */
    bool DequantizeOutput();

   private:
    ANeuralNetworksTFLite* mTFLite = nullptr;
    size_t mInputTensorByteSize = 0;
    size_t mOutputTensorByteSize = 0;
    size_t mNumOfClasses = 0;
    bool mApplySoftmax = false;
};

}  // namespace classifier
