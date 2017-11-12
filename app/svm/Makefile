COMPILE_ARG = -std=c++11 -lpthread -D_GLIBCXX_USE_NANOSLEEP -O3 -march=native
lr: svm.cpp
	mpic++ svm.cpp -o svm $(COMPILE_ARG)
	
clean:
	rm svm