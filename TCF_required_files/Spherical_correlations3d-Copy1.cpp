//
//  Created by Adélie Gorce on 04/01/2018.
//  Copyright © 2018 Adélie GORCE. All rights reserved.
//

/*
 Executable file to compute the spherical correlation function
 as defined in Gorce & Pritchard, 2019, MNRAS, 489, 1321–1337
 from files with real and imaginary parts of Fourier tranform
 of matter overdensity in 3D box
 */

#include "SC.h"
using namespace std;

static const int NDIM=3; //dimension of the box
static int nmodes=0;
fftwf_complex *field, *field_k;
fftwf_plan p;

/*
THIS VERSION INCLUDES LOOP OVER NORMS OF K AND ROTATIONS AROUND 3 AXES
*/

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
    double delta = 2.0*pi/L;
    for (int i=0; i<floor(N*0.5); i++) {
        y.push_back(i*delta);
    }
    for (int i=0; i<floor(N*0.5); i++) {
        y.push_back(-1*delta*(floor(N*0.5)-i));
    }
    return y;
}

static complex<double> epsilon_k[N][N][N];
static vector<double> k_x=k_axis();
const double delta_k = 2.0*pi/L;

complex<double> sigma_plus(int i, int j, int l, int i2, int j2, int l2, double r) {
    
    complex<double> B(0,0);
    int sx=0, sy=0, sz=0;
    double norm_p1=0,norm_p2=0,norm_p3=0, window_tot;
    
    if (i==N | j==N | l==N | i2==N | j2==N | l2==N) window_tot=0; //in main, loops start at i=0, but there is no symmetrical term (sigma(N-i)) as it would have index N-i = N which does not exist in overdensity array: need to remove this contribution from final sum by saying it is zero
    else {
        // bispectrum
        sx=i+i2; if (sx>N-1) sx-=N;
        sy=j+j2; if (sy>N-1) sy-=N;
        sz=l+l2; if (sz>N-1) sz-=N;
        B=epsilon_k[i2][j2][l2]*epsilon_k[i][j][l]*conj(epsilon_k[sx][sy][sz]);

        vector<double> window(NDIM);
        vector<double> p1(NDIM), p2(NDIM), p3(NDIM);

        p1[0]=k_x[i]+k_x[i2]*0.5+s3*k_x[j2]*0.5;
        p1[1]=k_x[j]-s3*k_x[i2]*0.5 + k_x[j2]*0.5;
        p1[2]=k_x[l]+k_x[l2];
        norm_p1=sqrt(pow(p1[0],2)+pow(p1[1],2)+pow(p1[2],2));
        if (norm_p1==0) window[0]=1.0;
        else window[0]=sin(norm_p1*r)/(norm_p1*r);

        p2[0]=k_x[i]+k_x[i2]*0.5-s3*0.5*k_x[l2];
        p2[1]=k_x[j]+k_x[j2];
        p2[2]=k_x[l]+s3*0.5*k_x[i2]+0.5*k_x[l2];
        norm_p2=sqrt(pow(p2[0],2)+pow(p2[1],2)+pow(p2[2],2));
        if (norm_p2==0) window[1]=1.0;
        else window[1]=sin(norm_p2*r)/(norm_p2*r);

        p3[0] = k_x[i]+k_x[i2];
        p3[1] = k_x[j]+ 0.5*k_x[j2] + s3*0.5*k_x[l2];
        p3[2] = k_x[l] - s3*0.5*k_x[j2] + 0.5*k_x[l2];
        norm_p3=sqrt(pow(p3[0],2)+pow(p3[1],2)+pow(p3[2],2));
        if (norm_p3==0) window[2]=1.0;
        else window[2]=sin(norm_p3*r)/(norm_p3*r);
        // p vector

        window_tot=window[0]+window[1]+window[2];

        if (i==floor(N/2) | j==floor(N/2) | l==floor(N/2) | i2==floor(N/2) | j2==floor(N/2) | l2==floor(N/2)) B*=0.5; //N/2th term will be counted 2 times because of symmetries in main function so need to divide value by 2
    }
    return B*window_tot;
}

double add_sigma(int i, int j, int l, int i2, int j2, int l2, double r) {

    complex<double> sigma(0,0);                 
    int a_values[2]={i,N-i};
    int b_values[2]={j,N-j};
    int c_values[2]={l,N-l};
    int d_values[2]={i2,N-i2};
    int e_values[2]={j2,N-j2};
    int f_values[2]={l2,N-l2};
    for (int a : a_values) {
        for (int b : b_values) {
            for (int c : c_values) {
                for (int d : d_values) {
                    for (int e : e_values) {
                        for (int f : f_values) {
                            sigma+=sigma_plus(a, b, c, d, e, f, r);
                        }
                    }
                }
            }
        }
    }

    return real(sigma);

}

complex<double> sum_sigma(double r) {
    
    // Initialisation
    double sigma_real=0, sigma_imag=0;

    vector<int> v(NDIM);
    int u=1;
    double *k_norm, kni=0;
    vector< vector<int> > k_ind;
    k_norm=(double *) malloc(1*sizeof(double));
    k_norm = (double*) realloc(k_norm,1*sizeof(double));
    k_norm[0]=0;
    v.assign({0,0,0});
    k_ind.push_back(v);
    v.clear();
    for (int i=0; i<floor(N*0.5); i++) {
        for (int j=0; j<=i; j++) {
            for (int l=0; l<=j; l++) {
                kni=sqrt(pow(k_x[i],2)+pow(k_x[j],2)+pow(k_x[l],2));
                if ((kni<=pi/r) && (kni!=0)) {
                    k_norm = (double*) realloc(k_norm,(u+1)*sizeof(double));
                    k_norm[u]=kni; 
                    v.assign({i,j,l});
                    k_ind.push_back(v);
                    //cout << u << " " << i << " " << j << " " << l << "  " << kni << " " << k_norm[u] << " " << v[0] << " " << v[1] << " " << v[2] <<endl;
                    v.clear();
                    u++;
                }
            }
        }
    }
    int ksize=u;
    // printf("\rr=%f, %i steps: %3d%% ", r, int(pow(ksize,2)), 0);

    int count=0, stage_step=2.;
    double stage=0;
    omp_set_num_threads(nthreads);
    #pragma omp parallel
    {

    vector<double> k(NDIM), q(NDIM);
    double norm_k=0, norm_q=0;
    nmodes = 0;

    #pragma omp for reduction(+:sigma_real,sigma_imag,nmodes)
    for (int i=0; i<ksize; i++) {
        for (int j=0; j<ksize; j++) {
            if ( k_norm[i]<= pi/r
            && k_norm[j]<= pi/r
            /*&& sqrt(pow(k_x[k_ind[i][0]]+k_x[k_ind[j][0]],2)+pow(k_x[k_ind[i][1]]+k_x[k_ind[j][1]],2)+pow(k_x[k_ind[i][2]]+k_x[k_ind[j][2]],2) ) <= pi/r */ 
            ) {

                sigma_real+= add_sigma(k_ind[i][0],k_ind[i][1],k_ind[i][2],k_ind[j][0],k_ind[j][1],k_ind[j][2],r);
                sigma_real+= add_sigma(k_ind[i][0],k_ind[i][1],k_ind[i][2],k_ind[j][0],k_ind[j][2],k_ind[j][1],r);
                sigma_real+= add_sigma(k_ind[i][0],k_ind[i][1],k_ind[i][2],k_ind[j][2],k_ind[j][1],k_ind[j][0],r);
                sigma_real+= add_sigma(k_ind[i][0],k_ind[i][1],k_ind[i][2],k_ind[j][1],k_ind[j][0],k_ind[j][2],r);

                sigma_real+= add_sigma(k_ind[i][1],k_ind[i][0],k_ind[i][2],k_ind[j][0],k_ind[j][1],k_ind[j][2],r);
                sigma_real+= add_sigma(k_ind[i][1],k_ind[i][0],k_ind[i][2],k_ind[j][1],k_ind[j][0],k_ind[j][2],r);
                sigma_real+= add_sigma(k_ind[i][1],k_ind[i][0],k_ind[i][2],k_ind[j][2],k_ind[j][1],k_ind[j][0],r);
                sigma_real+= add_sigma(k_ind[i][1],k_ind[i][0],k_ind[i][2],k_ind[j][0],k_ind[j][2],k_ind[j][1],r);

                sigma_real+= add_sigma(k_ind[i][2],k_ind[i][1],k_ind[i][0],k_ind[j][0],k_ind[j][1],k_ind[j][2],r);
                sigma_real+= add_sigma(k_ind[i][2],k_ind[i][1],k_ind[i][0],k_ind[j][1],k_ind[j][0],k_ind[j][2],r);
                sigma_real+= add_sigma(k_ind[i][2],k_ind[i][1],k_ind[i][0],k_ind[j][0],k_ind[j][2],k_ind[j][1],r);
                sigma_real+= add_sigma(k_ind[i][2],k_ind[i][1],k_ind[i][0],k_ind[j][2],k_ind[j][1],k_ind[j][0],r);

                sigma_real+= add_sigma(k_ind[i][0],k_ind[i][2],k_ind[i][1],k_ind[j][0],k_ind[j][1],k_ind[j][2],r);
                sigma_real+= add_sigma(k_ind[i][0],k_ind[i][2],k_ind[i][1],k_ind[j][1],k_ind[j][0],k_ind[j][2],r);
                sigma_real+= add_sigma(k_ind[i][0],k_ind[i][2],k_ind[i][1],k_ind[j][2],k_ind[j][1],k_ind[j][0],r);
                sigma_real+= add_sigma(k_ind[i][0],k_ind[i][2],k_ind[i][1],k_ind[j][0],k_ind[j][2],k_ind[j][1],r);

                nmodes+=16*6;

            }

		 //    #pragma omp critical 
		 //    {
		 //    	count++;
		 //    	stage=float(count)/float(pow(ksize,2))*100.;
   //              // cout << stage << " " << stage_step << " " << count <<endl;
		 //    	if (stage > stage_step) {
   //  			    int lpad = (int) (stage / 100. * PBWIDTH);
   //  			    int rpad = PBWIDTH - lpad;
   //  			    printf ("\r r=%f, %i steps: %3d%% [%.*s%*s]", r, int(ksize), int(stage), lpad, PBSTR, rpad, "");
	  //   		// cout << "|" <<flush;
	  //   		    stage_step+=2.;
		 //    	}
			// }

        }
    }

    }//end of parallelisation
    
    return complex<double>(sigma_real,sigma_imag)*pow(r/L,NDIM*1.5) ;

}

int main() {

    cout << "Dim in Fourier space " << N << " ; real space length " << L << " ; number of threads " << nthreads <<endl; 
    string file=string(filename_box)+string(".txt");

    field = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (N) * (N) * (N));
    field_k = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (N) * (N) * (N));

    //Reading overdensity files
    ifstream field_file ;
    field_file.open(file);
    if (!field_file.is_open() ) {cerr << "Unable to open file" <<endl; return -1;}
    cout << "\nFT files opened: reading..." <<endl ;
    for (int k=0; k<N; k++) {
        for (int i=0; i<N; i++) {
            for (int j=0; j<N; j++) {
                field_file >> field[i + N * (j + N * k)][0];
                field[i + N * j + N * N * k][1]=0;
            }
        }
    }
    cout << "Done reading file, computing FFT" <<endl;
    field_file.close();

    p = fftwf_plan_dft_3d(N, N, N, field, field_k, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);

    cout << field_k[12 + N * (12 + N * 12)][0] << " " << field_k[12 + N * (12 + N * 12)][1] <<endl;
    cout << field_k[(N-12) + N * ((N-12) + N * (N-12))][0] << " " << field_k[(N-12) + N * ((N-12) + N * (N-12))][1] <<endl;

    double abs=0;
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            for (int k=0; k<N; k++) {
                abs = sqrt( pow(field_k[i + N * (j + N * k)][0],2) + pow(field_k[i + N * (j + N * k)][1],2) );
                if ( abs<delta_k) epsilon_k[i][j][k]=0;
                else epsilon_k[i][j][k] = complex<double>(field_k[i + N * (j + N * k)][0],field_k[i + N * (j + N * k)][1])/abs;
            }
        }
    }

    string outfilename=string(filename_box)+string("_L")+to_string(int(L))+string("_3D_spherical_correlations.txt");
    ofstream outfile;
    outfile.open(outfilename);
    outfile << "#dim " << N <<endl;
    outfile << "# r Re[s(r)] Im[s(r)]" <<endl;

    vector<double> r=range(max(rmin,L/N),min(rmax,L*0.5),nbins);
    complex<double> l(0,0);
    auto start = chrono::steady_clock::now();
    cout << "Starting to compute the triangle correlation function for a range of correlation scales \n" <<endl;
    cout << " r / s(r) / Nmodes"  <<endl;
    for (int i=0; i<nbins; i++) {
    	printf("%.2f / ", r[i]);
        l=sum_sigma(r[i]);
        outfile << r[i] <<  " " << real(l) << " " << imag(l) << " " << nmodes << endl;
        // printf ("\33[2K");
	    printf ("\r%.2f  %.2f %.i \n", r[i],real(l),nmodes);
        // cout << " / " << l <<endl;// "     " << min_int(floor(N*0.5),floor(pi/(r[i]*delta_k))) <<endl;
    }
    outfile.close();

    auto end = chrono::steady_clock::now();
    auto howlong = end - start;
    cout << "Executing time: " << chrono::duration <double> (howlong).count()/60 << "min" <<endl;

    // ofstream timefile;
    // timefile.open("executing_times.txt", fstream::app);
    // timefile << "NDIM=" << NDIM << " N=" << N << " L=" << L << " threads: " << nthreads << " time = " << chrono::duration <double> (howlong).count()/60 << "min" <<endl;
    // timefile.close();

    // fftwf_destroy_plan(p);
    fftwf_free(field);
    fftwf_free(field_k);
    
    return 0;
}


