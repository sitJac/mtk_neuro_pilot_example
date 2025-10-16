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

#include <fstream>

#include "../logger/include/Logger.h"
#include "include/Utils.h"

#define TAG "Utils"

namespace utils {

bool ReadFileByLine(const std::string& file_path, std::vector<std::string>* result) {
    std::ifstream file(file_path);
    if (!file) {
        LOG_ERROR(TAG, "File %s not found", file_path.c_str());
        return false;
    }

    if (result == nullptr) {
        LOG_ERROR(TAG, "Null output vector");
        return false;
    }

    result->clear();
    std::string line;

    while (std::getline(file, line)) {
        result->push_back(line);
    }
    file.close();
    return true;
}

std::vector<uint8_t> ReadBinaryFile(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file) {
        LOG_ERROR(TAG, "Input bin not found");
        return {};
    }
    file.seekg(0, std::ios::end);
    auto input_file_size = file.tellg();
    std::vector<uint8_t> byte_buffer(input_file_size);

    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(byte_buffer.data()), input_file_size);
    file.close();
    return byte_buffer;
}
}  // namespace utils
