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

#include <chrono>
#include <cstdio>
#include <iomanip>
#include <ostream>
#include <vector>

namespace profiler {

class EventProfiler;

class NpProfiler {
   public:
    static NpProfiler& GetInstance() {
        static NpProfiler instance;
        return instance;
    }
    void Reg(const EventProfiler* prof) { mProfilers.push_back(prof); }
    int Enter() { return mIndent++; }
    void Exit() { mIndent--; }
    std::ostream& Print(std::ostream& o, int fnwidth) const;
    std::ostream& Log(std::ostream& o, const char* format, ...) const;
    NpProfiler(const NpProfiler&) = delete;
    NpProfiler(NpProfiler&&) = delete;
    NpProfiler& operator=(const NpProfiler&) = delete;

   private:
    std::vector<const EventProfiler*> mProfilers;
    int mIndent;
    NpProfiler() {}
    const std::string FormatTime(double v) const {
        char buf[1024];
        if (v < 1e-6) {
            sprintf(buf, "%8.2lfns", 1e9 * v);
        } else if (v < 1e-3) {
            sprintf(buf, "%8.2lfus", 1e6 * v);
        } else if (v < 1) {
            sprintf(buf, "%8.2lfms", 1e3 * v);
        } else {
            sprintf(buf, "%8.2lfs ", v);
        }
        return std::string(buf);
    }
};

inline const NpProfiler &GetInstance() { return NpProfiler::GetInstance(); }

class EventProfiler {
   public:
    EventProfiler(const char* fn, const int line, const char* func)
        : mCalls(0),
          mTotalTime(0),
          mDepth(0),
          mFileName(fn),
          mPrettyFunction(func),
          mLineNumber(line) {
        NpProfiler::GetInstance().Reg(this);
    }
    void Enter() {
        mCalls++;
        if (mDepth > 0) {
            double elapsed = Stop();
            mTotalTime += elapsed;  // suspend
            mMaxTime = elapsed > mMaxTime ? elapsed : mMaxTime;
            mMinTime = elapsed < mMaxTime ? elapsed : mMinTime;
        } else {
            mIndent = NpProfiler::GetInstance().Enter();
        }
        mDepth++;
        Start();
    }
    void Exit() {
        double elapsed = Stop();
        mTotalTime += elapsed;
        mMaxTime = elapsed > mMaxTime ? elapsed : mMaxTime;
        mMinTime = mMinTime == 0 ? elapsed : elapsed < mMinTime ? elapsed : mMinTime;
        mDepth--;
        if (mDepth > 0) {
            Start();  // resume
        } else {
            NpProfiler::GetInstance().Exit();
        }
    }
    const std::string &Filename() const { return mFileName; }
    int LineNumber() const { return mLineNumber; }
    const std::string Function() const { return std::string(mIndent, ' ') + mPrettyFunction; }
    int Calls() const { return mCalls; }
    double TotalTime() const { return mTotalTime; }
    double MinTime() const { return mMinTime; }
    double MaxTime() const { return mMaxTime; }

   private:
    std::chrono::high_resolution_clock::time_point mStartPoint;
    int mCalls = 0;
    double mTotalTime = 0;
    double mMinTime = 0;
    double mMaxTime = 0;
    int mDepth = 0;
    int mIndent = 0;
    void Start() { mStartPoint = std::chrono::high_resolution_clock::now(); }
    double Stop() {
        std::chrono::duration<double> elapsed =
            (std::chrono::high_resolution_clock::now() - mStartPoint);
        return (1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
    }
    const std::string mFileName;
    const std::string mPrettyFunction;
    const int mLineNumber;
};

template <typename T>
void profile_gate(const char* fn, const int line, const char* func, bool enter, T*) {
    static EventProfiler prof(fn, line, func);
    if (enter)
        prof.Enter();
    else
        prof.Exit();
}

inline std::ostream& NpProfiler::Log(std::ostream& o, const char* format, ...) const {
    va_list args;
    va_start(args, format);
    static char buffer[1024];
    vsprintf(buffer, format, args);
    o << buffer;
    va_end(args);
    return o;
}

inline std::ostream& NpProfiler::Print(std::ostream& o, int fnwidth = 60) const {
    o << std::endl
      << "Profiling Summary:" << std::endl
      << "-----------------------------------------------------------------------------------------"
         "--------"
      << std::endl;
    o << std::internal << std::setw(fnwidth) << "Function" << std::setw(10) << "Calls"
      << std::setw(15) << "Total time" << std::setw(10) << "Avg time" << std::setw(10) << "Min time"
      << std::setw(10) << "Max time" << std::endl
      << "-----------------------------------------------------------------------------------------"
         "--------"
      << std::endl;
    for (const EventProfiler* p : mProfilers) {
        o << std::internal << std::setw(fnwidth) << p->Function() << std::setw(10) << p->Calls()
          << std::setw(15) << FormatTime(p->TotalTime()) << std::setw(10)
          << FormatTime(p->TotalTime() / p->Calls()) << std::setw(10) << FormatTime(p->MinTime())
          << std::setw(10) << FormatTime(p->MaxTime()) << std::endl;
    }
    return o;
}

inline std::ostream& operator<<(std::ostream& o, const NpProfiler& prof) {
    return prof.Print(o, 120);
}

}  // namespace profiler

#define PROFILE_ME_AS(name)                                             \
    struct __prof_data {                                                \
        bool early_exit;                                                \
        __prof_data(const char* fn, const int line, const char* func) { \
            early_exit = false;                                         \
            profiler::profile_gate(fn, line, func, true, this);         \
        }                                                               \
        void Stop() {                                                   \
            early_exit = true;                                          \
            profiler::profile_gate(nullptr, -1, nullptr, false, this);  \
        }                                                               \
        ~__prof_data() {                                                \
            if (!early_exit) Stop();                                    \
        }                                                               \
    } __helper_var(__FILE__, __LINE__, name)
#define PROFILE_ME PROFILE_ME_AS(__PRETTY_FUNCTION__)
#define PROFILE_END __helper_var.Stop()
