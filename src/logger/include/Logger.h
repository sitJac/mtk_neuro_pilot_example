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

#include <cstdarg>

namespace logging {

enum LogSeverity {
  LOG_VERBOSE = 0,
  LOG_INFO = 1,
  LOG_WARNING = 2,
  LOG_ERROR = 3,
};

class Logger {
public:
    // Logging hook that takes variadic args.
    static void Log(LogSeverity severity, const char* tag, const char* format, ...);

    // Logging hook that takes a formatted va_list.
    static void LogFormatted(LogSeverity severity, const char* tag,
                            const char* format, va_list args);
private:
    static const char* GetSeverityName(LogSeverity severity);
};

}  // namespace logging

#define LOG_PROD(severity, tag, format, ...) \
    logging::Logger::Log(severity, tag, format, ##__VA_ARGS__);

#define LOG_VERBOSE(tag, format, ...)                               \
    LOG_PROD(logging::LOG_VERBOSE, tag, format, ##__VA_ARGS__);

#define LOG_VERBOSE_IF(cond, tag, format, ...)                      \
    do {                                                            \
        if (!cond) break;                                           \
        LOG_PROD(logging::LOG_VERBOSE, tag, format, ##__VA_ARGS__); \
    } while (false)

#define LOG_INFO(tag, format, ...)                               \
    LOG_PROD(logging::LOG_INFO, tag, format, ##__VA_ARGS__);

#define LOG_INFO_IF(cond, tag, format, ...)                      \
    do {                                                         \
        if (!cond) break;                                        \
        LOG_PROD(logging::LOG_INFO, tag, format, ##__VA_ARGS__); \
    } while (false)

#define LOG_WARN(tag, format, ...)                               \
    LOG_PROD(logging::LOG_WARNING, tag, format, ##__VA_ARGS__);

#define LOG_ERROR(tag, format, ...)                              \
    LOG_PROD(logging::LOG_ERROR, tag, format, ##__VA_ARGS__);