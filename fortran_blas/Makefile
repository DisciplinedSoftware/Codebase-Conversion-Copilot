FC = gfortran
FFLAGS = -O2
LIBS = -lblas

all: bench

bench: bench.f90
	$(FC) $(FFLAGS) -o bench bench.f90 $(LIBS)

clean:
	rm -f bench
