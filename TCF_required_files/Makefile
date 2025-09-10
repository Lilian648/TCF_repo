LDFLAGS= -lfftw3f_omp -lfftw3f -lm -L/usr/lib
CPPFLAGS=-I/usr/include/
CC=g++ -std=c++11 -fopenmp

2d: Spherical_correlations2d.cpp 
	$(CC) $(CPPFLAGS) -o ./SC_2d.o ./Spherical_correlations2d.cpp  $(LDFLAGS) 

3d: Spherical_correlations3d.cpp 
	$(CC) $(CPPFLAGS) -o ./SC_3d.o ./Spherical_correlations3d.cpp  $(LDFLAGS)

2d_cross: Spherical_correlations2d.cpp 
	$(CC) $(CPPFLAGS) -o ./CC_2d.o ./Cross_correlations2d.cpp  $(LDFLAGS) 

clean:
	rm -rf *.o
	rm -rf *~
