CXX=g++

default: libsamplers.a

libsamplers.a: samplers.o
	ar rcs $@ $^
    
samplers.o: samplers.cpp samplers.hpp
	$(CXX) -c $< -O3 -std=c++11 -fPIC

clean:
	rm *.o *.a
