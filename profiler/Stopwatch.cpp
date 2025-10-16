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

#include "include/Stopwatch.h"

#define TAG "Stopwatch"

namespace profiler {

Stopwatch::Stopwatch() { Reset(); }

void Stopwatch::Start() {
    if (!mRunning) {
        mStartPoint = std::chrono::high_resolution_clock::now();
        mRunning = true;
    }
}

void Stopwatch::Stop() {
    if (mRunning) {
        mEndPoint = std::chrono::high_resolution_clock::now();
        mElapsed += (mEndPoint - mStartPoint);
        mRunning = false;
    }
}

double Stopwatch::Elapsed(Timeunit unit) {
    std::chrono::duration<double> elapsed;
    if (mRunning) {
        elapsed = (std::chrono::high_resolution_clock::now() - mStartPoint);
    } else {
        elapsed = mElapsed;
    }

    if (unit == Timeunit::MICROSECONDS) {
        return std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    } else if (unit == Timeunit::MILLISECONDS) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    } else if (unit == Timeunit::SECONDS) {
        return std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    }
    return -1;
}

void Stopwatch::Reset() {
    mStartPoint = {};
    mEndPoint = {};
    mElapsed = std::chrono::milliseconds::zero();
    mRunning = false;
}

}  // namespace profiler
