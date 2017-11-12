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

#ifndef _MODEL_HPP_
#define _MODEL_HPP_

/*
    This class Model is the base class for every
    machine learning applications implemented in
    our PS.
*/

#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include "../storage/include_storage.hpp"

class Model {
   private:
   public:
    Model() {}
    virtual double compute_loss(const DataSet &ds, const Parameter &params, int num_of_all_data,
                                int num_workers, double lambda) = 0;
								
    virtual void compute_full_gradient(const DataSet &ds, const Parameter &params,
                                       Gradient_Dense &g, const int &num_of_all_data) = 0;
									
    virtual void update(const DataSet &ds, std::uniform_int_distribution<> &u,
                                          std::default_random_engine &e, Parameter &params,
                                          const Gradient_Dense &full_grad, const double &lambda,
                                          const int &num_epoches, const double &rate) = 0;


};

#endif