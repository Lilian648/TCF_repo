//
//  Created by Adélie Gorce on 04/01/2018.
//  Copyright © 2018 Adélie GORCE. All rights reserved.
//

/*
 Executable file to compute the spherical correlation function
 as defined in Gorce & Pritchard, 2019, MNRAS, 489, 1321–1337
 from files with real and imaginary parts of Fourier tranform
 of matter field in 2D box
 */

#include "SC.h"
using namespace std;

static const int NDIM = 2; //dimension of the box
static int nmodes = 0; //number of modes contributing to each probed scale r
fftwf_complex *field, *field_k;
fftwf_plan p;

/* VERSION THAT DOES NOT COMPUTE THE IMAGINARY PART OF THE TRIANGLE CORRELATION FUNCTION */

const int min_int(int a, int b) {
    if (a>b) return b;
    else if (a <= b) return a;
    return 0;
}

vector<double> range(double min, double max, size_t N) {
    vector<double> range;
    double delta = (max-min)/double(N-1);
    for(int i=0; i<N; i++) {
        range.push_back(min + i*delta);
    }
    return range;
}

vector<double> k_axis() {
    vector<double> y;
    double delta = 2*pi/L;
    for (int i=0; i<floor(N*0.5); i++) {
        y.push_back(i*delta);
    }
    for (int i=0; i<floor(N*0.5); i++) {
        y.push_back(-1*delta*(floor(N*0.5)-i));
    }
    return y;
}

static complex<double> epsilon_k[N][N];
static vector<double> k_x=k_axis();
const double delta_k = 2.0*pi/L;

complex<double> sigma_plus(int i, int j, int i2, int j2, double r) {

    complex<double> B(0,0);
    int sx=0, sy=0;
    double norm_p=0, window;

    if (i==N | j==N | i2==N | j2==N) window=0; //in main, loops start at i=0, but there is no symmetrical term (sigma(N-i)) as it would have index N-i = N which does not exist in field array: need to remove this contribution from final sum by saying it is zero
    else {
        vector<double> p(NDIM);
        p[0]=k_x[i]+k_x[i2]*0.5+s3*0.5*k_x[j2];
        p[1]=k_x[j]-s3*k_x[i2]*0.5+0.5*k_x[j2];
        sx=i+i2; if (sx>N-1) sx-=N;
        sy=j+j2; if (sy>N-1) sy-=N;
        // bispectrum
        B=epsilon_k[i2][j2]*epsilon_k[i][j]*conj(epsilon_k[sx][sy]);
        // p vector
        norm_p=sqrt(pow(p[0],2)+pow(p[1],2));
	//window function
        window=jn(0,norm_p*r);//Bessel function of the first kind order zero
        if (i==floor(N*0.5) | j==floor(N*0.5) | i2==floor(N*0.5) | j2==floor(N*0.5)) B*=0.5; //N/2th term will be counted 2 times because of symmetries in main function so need to divide value by 2
    }
    return B*window;
}

complex<double> add_sigma(int i, int j, int i2, int j2, double r) {

    complex<double> sigma(0,0);                 
    sigma+=sigma_plus(i, j, i2, j2, r);

    sigma+=sigma_plus(N-i, j, i2, j2, r);
    sigma+=sigma_plus(N-i, N-j, i2, j2, r);
    sigma+=sigma_plus(N-i, j,N- i2, j2, r);
    sigma+=sigma_plus(N-i, j, i2, N-j2, r);
    sigma+=sigma_plus(N-i, N-j, N-i2, j2, r);
    sigma+=sigma_plus(N-i, N-j, i2,N- j2, r);
    sigma+=sigma_plus(N-i, j, N-i2, N-j2, r);

    sigma+=sigma_plus(i, N-j, i2, j2, r);
    sigma+=sigma_plus(i, N-j,N-i2, j2, r);
    sigma+=sigma_plus(i, N-j, i2,N- j2, r);
    sigma+=sigma_plus(i, N-j, N-i2, N-j2, r);

    sigma+=sigma_plus(i, j, N-i2, j2, r);
    sigma+=sigma_plus(i, j, i2, N-j2, r);
    sigma+=sigma_plus(i, j, N-i2, N-j2, r);

    sigma+=sigma_plus(N-i, N-j, N-i2, N-j2, r);

    return sigma;

}

complex<double> sum_sigma(double r) {
// double sum_sigma(double r) {
 
    // Initialisation
    //int max_loop=min_int(floor(N*0.5),floor(pi/(r*delta_k))); //max value of loop index : k,q <= pi/r condition
    double sigma_real=0,sigma_imag=0;
    
    vector<int> v(NDIM);
    int u=1;
    double *k_norm, kni=0;
    vector< vector<int> > k_ind;
    k_norm=(double *) malloc(1*sizeof(double));
    k_norm = (double*) realloc(k_norm,1*sizeof(double));
    k_norm[0]=0;
    v.assign({0,0});
    k_ind.push_back(v);
    v.clear();
    for (int i=0; i<floor(N*0.5); i++) {
        for (int j=0; j<=i; j++) {
            kni=sqrt(pow(k_x[i],2)+pow(k_x[j],2));
            if ((kni<=pi/r) && (kni!=0)) {
                k_norm = (double*) realloc(k_norm,(u+1)*sizeof(double));
                k_norm[u]=kni; 
                v.assign({i,j});
                k_ind.push_back(v);
                //cout << u << " " << i << " " << j << " " << kni << " " << k_norm[u] << " " << v[0] << " " << v[1] <<endl;
                v.clear();
                u++;}
        }
    }
    int ksize=u;

    nmodes=0;
    omp_set_num_threads(nthreads);
    #pragma omp parallel 
    {

    double norm_k=0, norm_q=0;
    vector<double> k(NDIM), q(NDIM);

    #pragma omp for reduction (+: sigma_real, nmodes)
    for (int i=0; i<ksize; i++) {
        for (int j=0; j<ksize; j++) {
            if ( k_norm[i]<= pi/r
            && k_norm[j]<= pi/r ) {
            //&& sqrt(pow(k_x[k_ind[i][0]]+k_x[k_ind[j][0]],2)+pow(k_x[k_ind[i][1]]+k_x[k_ind[j][1]],2) ) <= pi/r) {
                sigma_real+= real(add_sigma(k_ind[i][0],k_ind[i][1],k_ind[j][0],k_ind[j][1],r));
                sigma_real+= real(add_sigma(k_ind[i][1],k_ind[i][0],k_ind[j][0],k_ind[j][1],r));
                sigma_real+= real(add_sigma(k_ind[i][0],k_ind[i][1],k_ind[j][1],k_ind[j][0],r));
                sigma_real+= real(add_sigma(k_ind[i][1],k_ind[i][0],k_ind[j][1],k_ind[j][0],r));
                nmodes += int(4*16);
            }
        }
    }
    sigma_imag=0;
	    
    } //end of parallelisation

    complex<double> sigma=complex<double>(sigma_real,sigma_imag);

    return sigma*pow(r/L,NDIM*1.5);

}


int main() {


    cout << "Dim in Fourier space " << N << " ; real space length " << L << " ; number of threads " << nthreads <<endl; 

    string file=string(filename_box)+string(".txt");

    field = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (N) * (N));
    field_k = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * N * (N));

    //Reading field files
    ifstream field_file ;
    field_file.open(file);
    if (!field_file.is_open() ) {cerr << "Unable to open file" <<endl; return -1;}
    cout << "Field file opened: reading..." <<endl ;
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            field_file >> field[i + N * j][0];
            field[i + N * j][1]=0;
            // cout << field[i + N * j][0] <<flush;
        }
    }
    cout << "Done reading file, computing FFT" <<endl;
    field_file.close();

    p = fftwf_plan_dft_2d(N, N, field, field_k, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);

    // check
    cout << field_k[12 + N * 12][0] << " " << field_k[12 + N * 12][1] <<endl;
    cout << field_k[(N-12) + N * (N-12)][0] << " " << field_k[(N-12) + N * (N-12)][1] <<endl;
    double abs=0;
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            abs = sqrt( pow(field_k[i + N * j][0],2) + pow(field_k[i + N * j][1],2) );
            if ( abs<delta_k) epsilon_k[i][j]=0;
            else epsilon_k[i][j] = complex<double>(field_k[i + N * j][0],field_k[i + N * j][1])/abs;
        }
    }

    string outfilename = string(filename_box)+string("_L")+to_string(int(L))+string("_spherical_correlations.txt");
    ofstream outfile;
    outfile.open(outfilename);
    outfile << "# dim " << N << endl;
    outfile << "# r Re[s(r)] Im[s(r)] N_modes" <<endl;

    vector<double> r=range(max(rmin,L/N),min(rmax,L*0.5),nbins);
    auto start = chrono::steady_clock::now();
    cout << " r / s(r) / Nmodes"  <<endl;
    for (int i=0; i<nbins; i++) {
        cout << r[i] << " / " <<flush;
        complex<double> l=sum_sigma(r[i]);
        outfile << r[i] << " " << real(l) << " " << imag(l) << " " << nmodes << endl;//<< real(l) << " " << imag(l) <<endl;
        cout << l << " " << nmodes <<endl;
    }
    outfile.close();

    auto end = chrono::steady_clock::now();
    auto howlong = end - start;
    cout << "Executing time: " << chrono::duration <double> (howlong).count()/60 << "min" <<endl;
    
    // fftwf_destroy_plan(p);
    fftwf_free(field);
    fftwf_free(field_k);

    return 0;

}
