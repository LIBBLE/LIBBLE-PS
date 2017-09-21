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

#ifndef _VECTOR_OPERATION_HPP_
#define _VECTOR_OPERATION_HPP_

#include <vector>

// add vec2 onto vec1
void vector_add(std::vector<double> &vec1, const std::vector<double> &vec2) {
    for (int i = 0; i < vec1.size(); i++) vec1[i] += vec2[i];
}

void vector_add(std::vector<int> &vec1, const std::vector<int> &vec2) {
    for (int i = 0; i < vec1.size(); i++) vec1[i] += vec2[i];
}

void vector_divi(std::vector<double> &vec, const double &x) {
    for (int i = 0; i < vec.size(); i++) vec[i] /= x;
}

void vector_multi_add(std::vector<double> &vec1, const double &x, const std::vector<double> &vec2,
                      const double &y) {
    for (int i = 0; i < vec1.size(); i++) vec1[i] = vec1[i] * x + vec2[i] * y;
}

void vector_divi_add(std::vector<double> &vec1, const double &x, const std::vector<double> &vec2,
                     const double &y) {
    for (int i = 0; i < vec1.size(); i++) vec1[i] = vec1[i] / x + vec2[i] * y;
}

void vector_sub(std::vector<double> &vec1, const std::vector<double> &vec2) {
    for (int i = 0; i < vec1.size(); i++) vec1[i] -= vec2[i];
}

double vector_multi(const std::vector<double> &vec1, const std::vector<double> &vec2) {
    double result = 0;
    for (int i = 0; i < vec1.size(); i++) result += vec1[i] * vec2[i];
    return result;
}

#endif