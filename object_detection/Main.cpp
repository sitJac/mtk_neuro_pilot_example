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

#include <getopt.h>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "../profiler/include/Profiler.h"
#include "../utils/include/CommonDef.h"
#include "DetectionEngine.h"
#include "Recognition.h"

#define TAG "MainApp"

#define MODEL_PATH "/data/local/tmp/detect.tflite"
#define INPUT_BIN_PATH "/data/local/tmp/voc_boat_pi_300300.bin"
#define LABEL_PATH "/data/local/tmp/labelmap.txt"

struct Settings {
    bool profiling = false;
    bool allow_fp16 = false;
    int loop_count = 1;
    float input_mean = DETECT_INPUT_MEAN;
    float input_std = DETECT_INPUT_STD;
    std::string model_path = MODEL_PATH;
    std::string input_bin_path = INPUT_BIN_PATH;
    std::string labels_path = LABEL_PATH;
    float threshold = DETECT_THRESHOLD;
    bool verbose = false;
};

bool run_inference(Settings* s) {
    bool ret = false;
    std::size_t input_file_size = 0;
    uint8_t* byte_buffer = nullptr;
    float* float_buffer = nullptr;
    const void* input_ptr = nullptr;
    std::size_t input_size = 0;
    std::vector<recognition::Recognition> results;

    LOG_INFO(TAG, "Construct engine with model");
    detection::DetectionEngine engine(s->model_path.c_str(), s->labels_path.c_str(), s->allow_fp16);

    engine.SetThreshold(s->threshold);
    engine.SetVerboseLog(s->verbose);
    engine.SetLoopCount(s->loop_count);

    do {
        std::ifstream file(s->input_bin_path);
        if (!file) {
            LOG_ERROR(TAG, "Input bin not found");
            break;
        }

        // Read file into buffer
        file.seekg(0, std::ios::end);
        input_file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        byte_buffer = new uint8_t[input_file_size];
        if (byte_buffer == nullptr) {
            LOG_ERROR(TAG, "Fail to allocate buffer to read input bin");
            break;
        }
        file.read(reinterpret_cast<char*>(byte_buffer), input_file_size);
        file.close();

        detection::ModelDataType model_type = engine.GetModelInputDataType();

        // Tranform byte data to float value
        if (model_type == detection::ModelDataType::TYPE_FLOAT_32) {
            float_buffer = new float[input_file_size * sizeof(float)];
            if (float_buffer == nullptr) {
                LOG_ERROR(TAG, "Fail to allocate buffer to process input bin");
                break;
            }
            for (auto i = 0; i < input_file_size; i++) {
                float_buffer[i] = (byte_buffer[i] - s->input_mean) / s->input_std;
            }
            input_ptr = reinterpret_cast<const void*>(float_buffer);
            input_size = input_file_size * sizeof(float);
        } else if (model_type == detection::ModelDataType::TYPE_UINT8) {
            input_ptr = reinterpret_cast<const void*>(byte_buffer);
            input_size = input_file_size;
        } else {
            LOG_ERROR(TAG, "Unsupported model data type");
            break;
        }

        LOG_INFO(TAG, "Inference with detection engine");
        // Inference with detect engine
        ret = engine.Inference(input_ptr, input_size, results);

        if (ret) {
            // Print the results
            LOG_INFO(TAG, "Detections:");
            for (auto const& i : results) {
                LOG_INFO(TAG, "    %s", i.GetTitle().c_str());
                LOG_INFO(TAG, "    - confidence: %f", i.GetConfidence());
                LOG_INFO(TAG, "    - rectangular window(left/top/right/bottom):%f/%f/%f/%f",
                         i.GetRect().left, i.GetRect().top, i.GetRect().right, i.GetRect().bottom);
            }
        }
    } while (false);

    if (byte_buffer != nullptr) {
        delete [] byte_buffer;
    }

    if (float_buffer != nullptr) {
        delete [] float_buffer;
    }

    return ret;
}

Settings process_options(int argc, char** argv) {
    Settings s;
    int c = 0;
    static struct option long_options[] = {{"allow_fp16", required_argument, nullptr, 'f'},
                                           {"count", required_argument, nullptr, 'c'},
                                           {"image_bin", required_argument, nullptr, 'i'},
                                           {"labels", required_argument, nullptr, 'l'},
                                           {"tflite_model", required_argument, nullptr, 'm'},
                                           {"profiling", required_argument, nullptr, 'p'},
                                           {"input_mean", required_argument, nullptr, 'b'},
                                           {"input_std", required_argument, nullptr, 's'},
                                           {"threshold", required_argument, nullptr, 't'},
                                           {"verbose", required_argument, nullptr, 'v'},
                                           {nullptr, 0, nullptr, 0}};

    while (1) {
        /* getopt_long stores the option index here. */
        int option_index = 0;
        c = getopt_long(argc, argv, "b:c:f:i:l:m:p:t:s:v:", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1) break;

        switch (c) {
            case 'b':
                s.input_mean = strtod(optarg, nullptr);
                break;
            case 'c':
                s.loop_count = strtol(optarg, nullptr, 10);
                break;
            case 'f':
                s.allow_fp16 = strtol(optarg, nullptr, 10);
                break;
            case 'i':
                s.input_bin_path = optarg;
                break;
            case 'l':
                s.labels_path = optarg;
                break;
            case 'm':
                s.model_path = optarg;
                break;
            case 'p':
                s.profiling = strtol(optarg, nullptr, 10);
                break;
            case 't':
                s.threshold = strtod(optarg, nullptr);
                break;
            case 's':
                s.input_std = strtod(optarg, nullptr);
                break;
            case 'v':
                s.verbose = strtol(optarg, nullptr, 10);
            default:
                break;
        }
    }
    return s;
}

int main(int argc, char** argv) {
    Settings s = process_options(argc, argv);
    if (run_inference(&s)) {
        if (s.profiling) {
            profiler::NpProfiler::GetInstance().Print(std::cout, 20);
        }
    }

    return 0;
}