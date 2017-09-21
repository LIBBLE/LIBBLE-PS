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

#ifndef _DATASET_HPP_
#define _DATASET_HPP_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "../util/include_util.hpp"
#include "DataPoint.hpp"

class DataSet {
   public:
    DataPoint *data;
    int num_rows;
    int num_cols;

    DataSet() {
        data = NULL;
        num_rows = -1;
        num_cols = -1;
    }

    ~DataSet() {
        if (data != NULL) delete[] data;
    }

    DataSet(const DataSet &d) = delete;

    DataSet &operator=(const DataSet &d) = delete;

    int get_num_rows() { return num_rows; }

    int get_num_cols() { return num_cols; }

    int read_from_file(std::string data_file, int id, int worker_num, int real_num_cols) {
        num_rows = 0;
        std::string buf;
        int file_count = 0;
        while (++file_count) {
            std::string data_files =
                data_file + "_/part" + std::to_string(id + ((file_count - 1) * worker_num));
            std::ifstream fin(data_files.c_str());
            if (!fin) break;
            while (getline(fin, buf)) {
                num_rows++;
            }
            fin.close();
        }

        data = new DataPoint[num_rows];
        file_count = 0;
        int row_count = 0;
        num_cols = 0;

        while (++file_count) {
            std::string data_files =
                data_file + "_/part" + std::to_string(id + ((file_count - 1) * worker_num));
            std::ifstream fin(data_files.c_str());
            if (!fin) break;
            while (getline(fin, buf)) {
                char str0[] = " :";

                char *result = strtok((char *)buf.c_str(), str0);

                if ((strcmp(result, "1") == 0) || (strcmp(result, "+1") == 0) ||
                    (strcmp(result, "1.0") == 0) || (strcmp(result, "+1.0") == 0))
                    data[row_count].label = 1.0;
                else
                    data[row_count].label = -1.0;

                while (result = strtok(NULL, str0)) {
                    // key start from 0
                    data[row_count].key.push_back(atoi(result) - 1);
                    result = strtok(NULL, str0);
                    data[row_count].value.push_back(atof(result));
                }
                if (data[row_count].key[data[row_count].key.size() - 1] > num_cols) {
                    num_cols = data[row_count].key[data[row_count].key.size() - 1];
                }

                row_count++;
            }
            fin.close();
        }
        // start from 0, so add 1
        num_cols++;
        std::cout << "Worker " << id << ": examples:" << num_rows << ",features:" << num_cols << "("
                  << real_num_cols - 1 << ")" << std::endl;
        num_cols = real_num_cols;
        for (int i = 0; i < num_rows; i++) {
            data[i].key.push_back(num_cols - 1);
            data[i].value.push_back(1.0);
        }

        return 0;
    }

    void count_c_num(std::vector<int> &c) {
        for (int i = 0; i < num_rows; i++) {
            for (auto &x : data[i].key) c[x]++;
        }
    }
};

#endif