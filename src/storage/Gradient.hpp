/**
        * Copyright (c) 2017 LIBBLE team supervised by Dr. Wu-Jun LI at Nanjing University.
        * All Rights Reserved.
        * Licensed under the Apache License, Version 2.0 (the "License");
        * you may not use this file except in compliance with the License.
        * You may obtain a copy of the License at
        *
        * http://www.apache.org/licenses/LICENSE-2.0
        *
        * Unless required by applicable law or agreed to in writing, software
        * distributed under the License is distributed on an "AS IS" BASIS,
        * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        * See the License for the specific language governing permissions and
        * limitations under the License. */

#ifndef _GRADIENT_HPP_
#define _GRADIENT_HPP_

#include <cassert>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "../util/include_util.hpp"

class Gradient_Dense {
   public:
    std::vector<double> gradient;

    Gradient_Dense() {}

    Gradient_Dense(const Gradient_Dense &g) = delete;

    Gradient_Dense &operator=(const Gradient_Dense &g) = delete;

    void resize(int s) { gradient.resize(s); }

    void reset() {
        for (auto &x : gradient) x = 0;
    }
};

// sparse gradient, to do
class Gradient_Sparse {
   public:
    std::vector<int> key;
    std::vector<double> value;

    void resize(int s) {
        key.resize(s);
        value.resize(s);
    }

    // to do
};

#endif