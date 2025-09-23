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

#include "../../logger/include/Logger.h"

#define UNUSED(x) (void)(x)

#define RETURN_FALSE_IF_NP_ERROR(code)                                  \
do {                                                                    \
    const auto _code = (code);                                          \
        if (_code != ANEURALNETWORKS_NO_ERROR) {                        \
            LOG_ERROR(TAG, "NeuroPilot API error (%d, %s, line %d)",    \
                    _code, __FILE__, __LINE__);                         \
            return false;                                               \
        }                                                               \
  } while (0)

#define RETURN_IF_NP_ERROR(code)                                        \
do {                                                                    \
    const auto _code = (code);                                          \
        if (_code != ANEURALNETWORKS_NO_ERROR) {                        \
            LOG_ERROR(TAG, "NeuroPilot API error (%d, %s, line %d)",    \
                    _code, __FILE__, __LINE__);                         \
            return;                                                     \
        }                                                               \
  } while (0)
