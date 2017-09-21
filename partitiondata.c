/*********************
g++ **.c -o ** -std=c++11
./partitiondata [filename] [number]
attention: it will make a new file named "data"+"_"
**********************/

#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <chrono>

int main(int argc, char **argv) {
	if(argc!=3){
		std::cout<<"argv error!\n";
		return -1;
	}
	
	std::string data_files =argv[1];
	int number = atoi(argv[2]);
	
	std::ifstream fin(data_files.c_str());
	if(!fin){
		std::cout<<"file open error!\n";
		return -1;
	}
	
	std::ofstream fout[number];
	for(int i=0;i<number;i++){
		std::string s = argv[1];
		std::string buffer = s + "_/part" + std::to_string(i+1);
		fout[i].open(buffer.c_str());
		if(!fout[i]){
			std::cout<<"file open error!\n";
			return -1;
		}
	}
	
	std::string buf;
	char str0[] = " ";
	int count = 0;
	while (getline(fin, buf)){
		char temp[buf.size()+1];
		buf.copy(temp,buf.size());
		temp[buf.size()] = '\0';
		strtok(temp, str0);//label
		char *result = NULL;
		result = strtok(NULL, str0);
		if(result == NULL)continue;
		fout[count%number]<<buf<<std::endl;
		count++;
	} 
	std::cout<<"number of data: "<<count<<std::endl;
	fin.close();
	for(int i = 0;i<number;i++){
		fout[i].close();
	}
	
	return 0;
} 

