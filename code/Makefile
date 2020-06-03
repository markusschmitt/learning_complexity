LIB_DIR = cpp

default: wolff_sampler

wolff_sampler: setup.py wolff_sampler.pyx $(LIB_DIR)/libwolffsampler.a
	python setup.py build_ext --inplace && rm -f wolff_sampler.cpp && rm -Rf build

$(LIB_DIR)/libwolffsampler.a:
	make -C $(LIB_DIR) libwolffsampler.a

clean:
	rm *.so
