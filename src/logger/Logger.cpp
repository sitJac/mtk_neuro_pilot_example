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

#include <cstdarg>

#include "include/Logger.h"

namespace logging {

void Logger::Log(LogSeverity severity, const char* tag, const char* format, ...) {
    va_list args;
    va_start(args, format);
    LogFormatted(severity, tag, format, args);
    va_end(args);
}

const char* Logger::GetSeverityName(LogSeverity severity) {
    switch (severity) {
        case LOG_VERBOSE:
            return "VERB";
        case LOG_INFO:
            return "INFO";
        case LOG_WARNING:
            return "WARN";
        case LOG_ERROR:
            return "ERROR";
    }
    return "<Unknown severity>";
}

}  // namespace logging
