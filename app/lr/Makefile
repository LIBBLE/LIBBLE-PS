COMPILE_ARG = -std=c++11 -lpthread -D_GLIBCXX_USE_NANOSLEEP -O3 -march=native
lr: lr.cpp
	mpic++ lr.cpp -o lr $(COMPILE_ARG)
	
clean:
	rm lr