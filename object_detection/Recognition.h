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

#include <string>

namespace recognition {

typedef struct Rect {
    float left;
    float top;
    float right;
    float bottom;
} Rect;

class Recognition {
public:
    Recognition(int class_id, const std::string& title, float confidence, const Rect& rect)
        : mClass(class_id), mTitle(title), mConfidence(confidence), mRect(rect) {}

    int GetClass() const { return mClass; }

    std::string GetTitle() const { return mTitle; }

    float GetConfidence() const { return mConfidence; }

    void Setrect(Rect& rect) { mRect = rect; }

    const Rect& GetRect() const { return mRect; }

private:
    int mClass;
    std::string mTitle;
    float mConfidence;
    Rect mRect;
};

}  // namespace recognition
