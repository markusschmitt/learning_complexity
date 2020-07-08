LIB_DIR = cpp

default: samplers

samplers: setup.py samplers.pyx $(LIB_DIR)/libsamplers.a
	python setup.py build_ext --inplace && rm -f wolff_sampler.cpp && rm -Rf build

$(LIB_DIR)/libsamplers.a:
	make -C $(LIB_DIR) libsamplers.a

clean:
	rm *.so
