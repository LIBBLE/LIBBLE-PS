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

#ifndef _ASSIST_FUNC_HPP_
#define _ASSIST_FUNC_HPP_

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)

#include <fstream>
#include <string>

void error(const char *loc, const char *msg) {
    std::cout << loc << ": " << msg << std::endl;
    exit(-1);
}

/* parse the argument */
int arg_parser(std::string str, int argc, char **argv) {
    int pos;
    for (pos = 0; pos < argc; pos++) {
        if (str.compare(argv[pos]) == 0) {
            return pos;
        }
    }
    return -1;
}

/* helps to find how many parameter a certain server holds */
int get_local_params_size(const int &n_cols, const int &n_servers, const int &server_id) {
    int x = n_cols / n_servers, y = n_cols % n_servers;
    if (y > 0) {
        if (server_id <= y)
            return x + 1;
        else
            return x;
    } else {
        return x;
    }
}

void write_file(std::string data_file, std::string info, double loss, double accuracy) {
    std::string output_file = data_file;
    std::ofstream fout(output_file.c_str(), std::ios::out | std::ios::app);
    fout.precision(15);
    fout << info << loss << " " << accuracy << std::endl;
    fout.close();
}

#endif