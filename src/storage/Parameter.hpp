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

#ifndef _PARAMETER_HPP_
#define _PARAMETER_HPP_

#include <cassert>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "../util/include_util.hpp"
#include "Gradient.hpp"

class Parameter {
   public:
    std::vector<double> parameter;
    Parameter() {}
    Parameter(const std::vector<double> &other_params) { parameter = other_params; }

    Parameter(const Parameter &p) = delete;

    Parameter &operator=(const Parameter &g) = delete;

    void resize(int s) { parameter.resize(s); }

    void reset() {
        for (auto &x : parameter) x = 0;
    }

    void parameter_random_init() {
        std::random_device rd;
        std::default_random_engine e(rd());
        std::uniform_real_distribution<> u(0, 1);
        for (auto &x : parameter) x = u(e);
    }

    void subs_gradient(const Gradient_Dense &g, const double &rate) {
        for (int i = 0; i < parameter.size(); i++) {
            parameter[i] -= g.gradient[i] * rate;
        }
    }

    void soft_threshold(double z) {
        for (auto &x : parameter) {
            if (x > z)
                x -= z;
            else if (x < -z)
                x += z;
            else
                x = 0;
        }
    }

    // get a slice of parameters
    std::vector<double> slice(int s, int e) {
        std::vector<double> slice_parameter;
        for (int i = s; i < e; i++) {
            slice_parameter.push_back(parameter[i]);
        }

        return slice_parameter;
    }

    std::vector<double> get_parameter() { return parameter; }

    void save_into_file(std::string data_file) {
        std::string output_file = data_file + "_output";
        std::ofstream fout(output_file.c_str());
        for (int i = 0; i < parameter.size(); i++) {
            fout << parameter[i] << " ";
        }
        fout << std::endl;
        fout.close();
    }
};

#endif