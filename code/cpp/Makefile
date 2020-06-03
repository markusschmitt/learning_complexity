CXX=g++-8

default: libwolffsampler.a

libwolffsampler.a: wolff_obc.o
	ar rcs $@ $^
    
wolff_obc.o: wolff_obc.cpp wolff_obc.hpp
	$(CXX) -c $< -O3 -std=c++11 -fPIC

clean:
	rm *.o *.a
