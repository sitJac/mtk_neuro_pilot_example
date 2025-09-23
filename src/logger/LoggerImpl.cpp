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

#include <android/log.h>
#include <cstdio>

#include "include/Logger.h"

namespace logging {
namespace {

int GetPlatformSeverity(LogSeverity severity) {
    switch (severity) {
        case LOG_VERBOSE:
            return ANDROID_LOG_VERBOSE;
        case LOG_INFO:
            return ANDROID_LOG_INFO;
        case LOG_WARNING:
            return ANDROID_LOG_WARN;
        case LOG_ERROR:
            return ANDROID_LOG_ERROR;
        default:
            return ANDROID_LOG_DEBUG;
    }
}

}  // namespace

void Logger::LogFormatted(LogSeverity severity, const char* tag,
                                 const char* format, va_list args) {
  // First log to Android's explicit log(cat) API.
  va_list args_for_android_log;
  va_copy(args_for_android_log, args);
  __android_log_vprint(GetPlatformSeverity(severity), tag, format, args);
  va_end(args_for_android_log);

  // Also print to stderr for standard console applications.
  fprintf(stderr, "[%s][%s]: ", GetSeverityName(severity), tag);
  vfprintf(stderr, format, args);
  fputc('\n', stderr);
}

}  // namespace logging
