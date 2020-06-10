CXX=g++-8

default: libwolffsampler.a

libwolffsampler.a: wolff.o
	ar rcs $@ $^
    
wolff_obc.o: wolff.cpp wolff.hpp
	$(CXX) -c $< -O3 -std=c++11 -fPIC

clean:
	rm *.o *.a
