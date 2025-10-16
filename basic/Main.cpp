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

#include <getopt.h>

#include "../utils/include/CommonDef.h"
#include "GenericClassifier.h"

#define TAG "GenericClassifierApp"

struct Settings {
    std::string model_path = "";
    std::string input_bin_path = "";
    std::string output_bin_path = "";
    bool print_model_info = false;
    bool enable_softmax = false;
};

void display_usage() {
    LOG_INFO(TAG,
             "\nMnist Classifier App\n"
             "--input, -i: string Path to the input bin file.\n"
             "--output, -o: string Path to the output bin file.\n"
             "--tflite_model, -m: string Path to the TFLite model file\n"
             "--model_info, -p: int (default=0) "
             "Set to 1 to print the TFLite model information\n"
             "--enable_softmax, -s: int (default=0) "
             "Set to 1 to apply softmax on the inference result\n");
}

Settings process_options(int argc, char** argv) {
    Settings s;
    int c = 0;
    static struct option long_options[] = {{"input", required_argument, nullptr, 'i'},
                                           {"output", required_argument, nullptr, 'o'},
                                           {"tflite_model", required_argument, nullptr, 'm'},
                                           {"model_info", required_argument, nullptr, 'p'},
                                           {"enable_softmax", required_argument, nullptr, 's'},
                                           {nullptr, 0, nullptr, 0}};

    while (1) {
        /* getopt_long stores the option index here. */
        int option_index = 0;
        c = getopt_long(argc, argv, "i:o:m:p:s:h:", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1) break;

        switch (c) {
            case 'i':
                s.input_bin_path = optarg;
                break;
            case 'o':
                s.output_bin_path = optarg;
                break;
            case 'm':
                s.model_path = optarg;
                break;
            case 'p':
                s.print_model_info = strtol(optarg, nullptr, 10);
                break;
            case 's':
                s.enable_softmax = strtol(optarg, nullptr, 10);
                break;
            case 'h':
            case '?':
                display_usage();
                exit(-1);
            default:
                break;
        }
    }
    return s;
}

int main(int argc, char** argv) {
    uint32_t predicted_class = 0;
    Settings s = process_options(argc, argv);
    classifier::GenericClassifier classifier(s.model_path);
    if (s.print_model_info) {
        classifier.PrintModelInformation();
    }

    classifier.EnableSoftmax(s.enable_softmax);

    if (classifier.Inference(s.input_bin_path, s.output_bin_path, &predicted_class)) {
        LOG_INFO(TAG, "Predicted class: %d", predicted_class);
    } else {
        LOG_ERROR(TAG, "Fail to predict class.")
    }
    return 0;
}
