#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/*
Here we demonstrate the calculation of single and double integrals on GPUs using CUDA
We simultaneously carry out 20 of the same integral, with different input parameters
The prefix h_ corresponds to host (namely, CPU)
The prefix d_ corresponds to device (namely, GPU)
Specifically, we evaluate the integrals:
\int_3^5  A \theta^2 + \exp(B\theta) d\theta			(1)
\int_3^5 \int_1^2  C \phi+D\theta^2 d\theta d\phi	(2)
A and B are constants stored in h_params_1d[] and d_params_1d[]
C and D are constants stored in h_params_2d[] and d_params_2d[]
The limits are stored in h_limits[] and d_limits[]
*/

__constant__ int d_number_of_integrals_1d,d_number_of_integrals_2d, d_datapoints, d_block_mult;

//The following two functions carry out a single integral (1)
__device__ double d_integrand1(double *d_params,double theta)
{
	double integrand=d_params[0]*theta*theta+exp(d_params[1]*theta);
	return integrand;
}

double h_integrand1(double *h_params,double theta)
{
	double integrand=h_params[0]*theta*theta+exp(h_params[1]*theta);
	return integrand;
}

__global__ void d_single_integral(double *d_params, double *d_limits, double *d_result, double *d_w, double *d_x)
{	
	//Calculate the integrand on each thread
	int node_num, integral_num;
	integral_num=threadIdx.x/d_datapoints;
	node_num=threadIdx.x % d_datapoints;
	integral_num+=blockIdx.x*d_block_mult;
	extern  __shared__ double d_result_temp[];	//Integrand is stored in shared memory
	double lim=d_limits[1]-d_limits[0];
	double theta=d_x[node_num]*lim+d_limits[0];
	double integrand=d_integrand1(d_params+integral_num*2,theta);
	d_result_temp[threadIdx.x]=integrand*d_w[node_num]*lim;

	//Perform a standard reduction to integrate
    for (int d = d_datapoints>>1; d > 0; d >>= 1)
    {
      __syncthreads(); 
      if (node_num<d)
	{
		d_result_temp[threadIdx.x]   += d_result_temp[threadIdx.x+d];
	}
    }
	__syncthreads();
    if (node_num==0) 
    {	
	d_result[integral_num]=d_result_temp[threadIdx.x];
    }
}

//Exact, analytic solution
double h_integral1_exact(double *h_params, double *h_limits)
{
	return h_params[0]*(h_limits[1]*h_limits[1]*h_limits[1]-h_limits[0]*h_limits[0]*h_limits[0])/3.0+(exp(h_params[1]*h_limits[1])-exp(h_params[1]*h_limits[0]))/h_params[1];
}


//Two functions to carry out double integral (2)
__device__ double d_integrand2(double *d_params,double theta,double phi)
{
	double integrand=d_params[0]*phi+d_params[1]*theta*theta;
	return integrand;
}

double h_integrand2(double *h_params,double theta,double phi)
{
	double integrand=h_params[0]*phi+h_params[1]*theta*theta;
	return integrand;
}

//Two-dimensional integral involves a loop over the first dimension
__global__ void d_double_integral(double *d_params, double *d_limits, double *d_result, double *d_w, double *d_x)
{	
	//Calculate the integrand on each thread
	int node_num, integral_num;
	integral_num=threadIdx.x/d_datapoints;
	node_num=threadIdx.x % d_datapoints;
	integral_num+=blockIdx.x*d_block_mult;
	extern  __shared__ double d_result_temp[];
	double lim1=d_limits[1]-d_limits[0];
	double lim2=d_limits[3]-d_limits[2];
	double theta=d_x[node_num]*lim1+d_limits[0];
	double phi, integrand, int_w_w=0.0;
	//Loop over first dimension
	for (int idx=0;idx<d_datapoints;idx++)
	{
		phi=d_x[idx]*lim2+d_limits[2];
		integrand=d_integrand2(d_params+integral_num*2,theta,phi);
		int_w_w+=integrand*d_w[idx];
	}
	d_result_temp[threadIdx.x]=int_w_w*d_w[node_num]*lim1*lim2;


	//Reduction over the second integral to obtain a value for the full double integral
    for (int d = d_datapoints>>1; d > 0; d >>= 1)
    {
      __syncthreads(); 
      if (node_num<d)
	{
		d_result_temp[threadIdx.x]   += d_result_temp[threadIdx.x+d];
	}
    }
	__syncthreads();
    if (node_num==0) 
    {	
	d_result[integral_num]=d_result_temp[threadIdx.x];
    }
}

//Exact, analytic solution
double h_integral2_exact(double *h_params, double *h_limits)
{
	return h_params[0]*(h_limits[3]*h_limits[3]-h_limits[2]*h_limits[2])*(h_limits[1]-h_limits[0])/2.0+h_params[1]*(h_limits[1]*h_limits[1]*h_limits[1]-h_limits[0]*h_limits[0]*h_limits[0])*(h_limits[3]-h_limits[2])/3.0;
}


void gauss_integration_setup_fromfile(int *datapoints, double **weights, double **x)
{
	FILE *INFILE; int idx;
	if ((INFILE=fopen("Gauss_Integration.txt", "r"))==NULL)
		{
		printf("Cannot open existing nodes/weights file! Error!\n");
		exit(2);
	       	}

	fscanf(INFILE,"%i", datapoints);
	*x=(double*)malloc(*datapoints*sizeof(double));		
	*weights=(double*)malloc(*datapoints*sizeof(double));
	for(idx=0;idx< *datapoints;idx++)
		{
		fscanf(INFILE,"%lf %lf", &x[0][idx], &weights[0][idx]);
		}
	fclose(INFILE);
}

void gauss_integration_setup32(int *datapoints, double **weights, double **x)
{
	*datapoints=32;
	*x=(double*)malloc(32*sizeof(double));		
	*weights=(double*)malloc(32*sizeof(double));
	x[0][0]=1.3680690752591596E-03;
	x[0][1]=7.1942442273659202E-03;
	x[0][2]=1.7618872206246805E-02;
	x[0][3]=3.2546962031130167E-02;
	x[0][4]=5.1839422116973843E-02;
	x[0][5]=7.5316193133715015E-02;
	x[0][6]=1.0275810201602886E-01;
	x[0][7]=1.3390894062985509E-01;
	x[0][8]=1.6847786653489233E-01;
	x[0][9]=2.0614212137961868E-01;
	x[0][10]=2.4655004553388526E-01;
	x[0][11]=2.8932436193468253E-01;
	x[0][12]=3.3406569885893617E-01;
	x[0][13]=3.8035631887393162E-01;
	x[0][14]=4.2776401920860185E-01;
	x[0][15]=4.7584616715613093E-01;
	x[0][16]=5.2415383284386907E-01;
	x[0][17]=5.7223598079139815E-01;
	x[0][18]=6.1964368112606838E-01;
	x[0][19]=6.6593430114106378E-01;
	x[0][20]=7.1067563806531764E-01;
	x[0][21]=7.5344995446611462E-01;
	x[0][22]=7.9385787862038115E-01;
	x[0][23]=8.3152213346510750E-01;
	x[0][24]=8.6609105937014474E-01;
	x[0][25]=8.9724189798397114E-01;
	x[0][26]=9.2468380686628515E-01;
	x[0][27]=9.4816057788302599E-01;
	x[0][28]=9.6745303796886994E-01;
	x[0][29]=9.8238112779375319E-01;
	x[0][30]=9.9280575577263397E-01;
	x[0][31]=9.9863193092474067E-01;

	weights[0][0]=3.5093050047349198E-03;
	weights[0][1]=8.1371973654528751E-03;
	weights[0][2]=1.2696032654631021E-02;
	weights[0][3]=1.7136931456510726E-02;
	weights[0][4]=2.1417949011113720E-02;
	weights[0][5]=2.5499029631187890E-02;
	weights[0][6]=2.9342046739268091E-02;
	weights[0][7]=3.2911111388180682E-02;
	weights[0][8]=3.6172897054423871E-02;
	weights[0][9]=3.9096947893535162E-02;
	weights[0][10]=4.1655962113473763E-02;
	weights[0][11]=4.3826046502202044E-02;
	weights[0][12]=4.5586939347882056E-02;
	weights[0][13]=4.6922199540401971E-02;
	weights[0][14]=4.7819360039637472E-02;
	weights[0][15]=4.8270044257364274E-02;
	weights[0][16]=4.8270044257363830E-02;
	weights[0][17]=4.7819360039637784E-02;
	weights[0][18]=4.6922199540401846E-02;
	weights[0][19]=4.5586939347881918E-02;
	weights[0][20]=4.3826046502201850E-02;
	weights[0][21]=4.1655962113473798E-02;
	weights[0][22]=3.9096947893534850E-02;
	weights[0][23]=3.6172897054424745E-02;
	weights[0][24]=3.2911111388180932E-02;
	weights[0][25]=2.9342046739267064E-02;
	weights[0][26]=2.5499029631188164E-02;
	weights[0][27]=2.1417949011113362E-02;
	weights[0][28]=1.7136931456510799E-02;
	weights[0][29]=1.2696032654631212E-02;
	weights[0][30]=8.1371973654529653E-03;
	weights[0][31]=3.5093050047351631E-03;
}

void gauss_integration_setup32_f(int *datapoints, float **weights, float **x)
{
	*datapoints=32;
	*x=(float*)malloc(32*sizeof(float));		
	*weights=(float*)malloc(32*sizeof(float));
	x[0][0]=1.3680690752591596E-03f;
	x[0][1]=7.1942442273659202E-03f;
	x[0][2]=1.7618872206246805E-02f;
	x[0][3]=3.2546962031130167E-02f;
	x[0][4]=5.1839422116973843E-02f;
	x[0][5]=7.5316193133715015E-02f;
	x[0][6]=1.0275810201602886E-01f;
	x[0][7]=1.3390894062985509E-01f;
	x[0][8]=1.6847786653489233E-01f;
	x[0][9]=2.0614212137961868E-01f;
	x[0][10]=2.4655004553388526E-01f;
	x[0][11]=2.8932436193468253E-01f;
	x[0][12]=3.3406569885893617E-01f;
	x[0][13]=3.8035631887393162E-01f;
	x[0][14]=4.2776401920860185E-01f;
	x[0][15]=4.7584616715613093E-01f;
	x[0][16]=5.2415383284386907E-01f;
	x[0][17]=5.7223598079139815E-01f;
	x[0][18]=6.1964368112606838E-01f;
	x[0][19]=6.6593430114106378E-01f;
	x[0][20]=7.1067563806531764E-01f;
	x[0][21]=7.5344995446611462E-01f;
	x[0][22]=7.9385787862038115E-01f;
	x[0][23]=8.3152213346510750E-01f;
	x[0][24]=8.6609105937014474E-01f;
	x[0][25]=8.9724189798397114E-01f;
	x[0][26]=9.2468380686628515E-01f;
	x[0][27]=9.4816057788302599E-01f;
	x[0][28]=9.6745303796886994E-01f;
	x[0][29]=9.8238112779375319E-01f;
	x[0][30]=9.9280575577263397E-01f;
	x[0][31]=9.9863193092474067E-01f;

	weights[0][0]=3.5093050047349198E-03f;
	weights[0][1]=8.1371973654528751E-03f;
	weights[0][2]=1.2696032654631021E-02f;
	weights[0][3]=1.7136931456510726E-02f;
	weights[0][4]=2.1417949011113720E-02f;
	weights[0][5]=2.5499029631187890E-02f;
	weights[0][6]=2.9342046739268091E-02f;
	weights[0][7]=3.2911111388180682E-02f;
	weights[0][8]=3.6172897054423871E-02f;
	weights[0][9]=3.9096947893535162E-02f;
	weights[0][10]=4.1655962113473763E-02f;
	weights[0][11]=4.3826046502202044E-02f;
	weights[0][12]=4.5586939347882056E-02f;
	weights[0][13]=4.6922199540401971E-02f;
	weights[0][14]=4.7819360039637472E-02f;
	weights[0][15]=4.8270044257364274E-02f;
	weights[0][16]=4.8270044257363830E-02f;
	weights[0][17]=4.7819360039637784E-02f;
	weights[0][18]=4.6922199540401846E-02f;
	weights[0][19]=4.5586939347881918E-02f;
	weights[0][20]=4.3826046502201850E-02f;
	weights[0][21]=4.1655962113473798E-02f;
	weights[0][22]=3.9096947893534850E-02f;
	weights[0][23]=3.6172897054424745E-02f;
	weights[0][24]=3.2911111388180932E-02f;
	weights[0][25]=2.9342046739267064E-02f;
	weights[0][26]=2.5499029631188164E-02f;
	weights[0][27]=2.1417949011113362E-02f;
	weights[0][28]=1.7136931456510799E-02f;
	weights[0][29]=1.2696032654631212E-02f;
	weights[0][30]=8.1371973654529653E-03f;
	weights[0][31]=3.5093050047351631E-03f;
}

int main(int argc, char *argv[])
{

	int number_of_integrals_1d=20, number_of_integrals_2d=20, h_block_mult=1;
	if (argc>1){number_of_integrals_1d=atoi(argv[1]);}
	if (argc>2){number_of_integrals_2d=atoi(argv[2]);}
	if (argc>3){h_block_mult=atoi(argv[3]);}

	//number of nodes for integration
	int h_datapoints;

	//Initialisation of GPU
	int device_count=0;
	cudaGetDeviceCount(&device_count);
	cudaSetDevice(0);	//Run on device 0 by default - can be changed if multiple GPUs etc are present
	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);

	//Declare, allocate and calculate nodes and weights for integration
	double *h_x,*h_w;
	gauss_integration_setup32(&h_datapoints,&h_w,&h_x);

	//Copy nodes and weights to GPU
	double *d_x, *d_w;
	cudaMemcpyToSymbol(d_block_mult,&h_block_mult,sizeof(h_block_mult));
	cudaMemcpyToSymbol(d_datapoints,&h_datapoints,sizeof(h_datapoints));
	cudaMalloc((void **)&d_x,sizeof(double)*h_datapoints);
	cudaMalloc((void **)&d_w,sizeof(double)*h_datapoints);
	cudaMemcpy(d_x,h_x,sizeof(double)*h_datapoints,cudaMemcpyHostToDevice);
	cudaMemcpy(d_w,h_w,sizeof(double)*h_datapoints,cudaMemcpyHostToDevice);

	//Allocate integral-specific constants, limits of integration
	int idx, idx1, idx2;
	double *h_params_1d, *h_params_2d, *d_params_1d, *d_params_2d, *h_lims_1d, *h_lims_2d, *d_lims_1d, *d_lims_2d, *h_result_1d, *h_result_2d, *d_result, h_theta, h_phi, h_lim, h_lim1, h_lim2;
	h_params_1d=(double*)malloc(sizeof(double)*number_of_integrals_1d*2);
	h_params_2d=(double*)malloc(sizeof(double)*number_of_integrals_2d*2);
	h_lims_1d=(double*)malloc(sizeof(double)*2);
	h_lims_2d=(double*)malloc(sizeof(double)*4);
	h_lims_1d[0]=3.0; h_lims_1d[1]=5.0;
	h_lims_2d[0]=3.0; h_lims_2d[1]=5.0; h_lims_2d[2]=1.0; h_lims_2d[3]=2.0;
	h_result_1d=(double*)malloc(sizeof(double)*number_of_integrals_1d*2);
	h_result_2d=(double*)malloc(sizeof(double)*number_of_integrals_2d*2);
	cudaMalloc((void **)&d_params_1d,sizeof(double)*number_of_integrals_1d*2);
	cudaMalloc((void **)&d_params_2d,sizeof(double)*number_of_integrals_2d*2);
	cudaMalloc((void **)&d_lims_1d,sizeof(double)*2);
	cudaMalloc((void **)&d_lims_2d,sizeof(double)*4);
	cudaMalloc((void **)&d_result,sizeof(double)*(number_of_integrals_1d+number_of_integrals_2d));
	cudaMemcpy(d_lims_1d,h_lims_1d,sizeof(double)*2,cudaMemcpyHostToDevice);
	cudaMemcpy(d_lims_2d,h_lims_2d,sizeof(double)*4,cudaMemcpyHostToDevice);

	//Set A, B, C D parameters of integrals to something non-trivial
	for(int idx=0;idx<number_of_integrals_1d;idx++)
		{
		h_params_1d[idx*2]=(rand() % 1000)/10000.0*(((double)idx)+1.0);
		h_params_1d[idx*2+1]=(rand() % 1000)/20000.0*((double)(idx*idx)+1.0);
		}
	for(int idx=0;idx<number_of_integrals_2d;idx++)
		{
		h_params_2d[idx*2]=(rand() % 10000)/10000.0*(((double)idx)+1.0);
		h_params_2d[idx*2+1]=(rand() % 10000)/20000.0*((double)(idx*idx)+1.0);
		}

	//Timing variables
	clock_t h_start_t, h_end_t;
	float d_time, h_time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Start integration
	cudaEventRecord(start);

	//Copy parameters
	cudaMemcpy(d_params_1d,h_params_1d,sizeof(double)*number_of_integrals_1d*2,cudaMemcpyHostToDevice);
	cudaMemcpy(d_params_2d,h_params_2d,sizeof(double)*number_of_integrals_2d*2,cudaMemcpyHostToDevice);
	//Carry out both integrals
	d_single_integral<<<number_of_integrals_1d/h_block_mult,h_datapoints*h_block_mult,h_datapoints*h_block_mult*sizeof(double),streams[0]>>>(d_params_1d,d_lims_1d,d_result,d_w,d_x);
	d_double_integral<<<number_of_integrals_2d/h_block_mult,h_datapoints*h_block_mult,h_datapoints*h_block_mult*sizeof(double),streams[1]>>>(d_params_2d,d_lims_2d,d_result+number_of_integrals_1d,d_w,d_x);
	//Copy result back when each stream is done
	cudaMemcpyAsync(h_result_1d+number_of_integrals_1d,d_result,sizeof(double)*number_of_integrals_1d,cudaMemcpyDeviceToHost,streams[0]);
	cudaMemcpyAsync(h_result_2d+number_of_integrals_2d,d_result+number_of_integrals_1d,sizeof(double)*number_of_integrals_2d,cudaMemcpyDeviceToHost,streams[1]);

	cudaEventRecord(stop);
	cudaEventElapsedTime(&d_time, start, stop);

	h_start_t=clock();
	//CPU integral evaluation
	h_lim=h_lims_1d[1]-h_lims_1d[0];
	h_lim1=h_lims_2d[1]-h_lims_2d[0];
	h_lim2=h_lims_2d[3]-h_lims_2d[2];
	for(idx=0;idx<number_of_integrals_1d;idx++)
		{
		h_result_1d[idx]=0.0;
		for(idx1=0;idx1<h_datapoints;idx1++)
			{
			h_theta=h_x[idx1]*h_lim+h_lims_1d[0];
			h_result_1d[idx]+=h_integrand1(h_params_1d+idx*2,h_theta)*h_w[idx1]*h_lim;
			}
		}

	for(idx=0;idx<number_of_integrals_2d;idx++)
		{
		h_result_2d[idx]=0.0;
		for(idx1=0;idx1<h_datapoints;idx1++)
			{
			h_theta=h_x[idx1]*h_lim1+h_lims_2d[0];
			for(idx2=0;idx2<h_datapoints;idx2++)
				{
				h_phi=h_x[idx2]*h_lim2+h_lims_2d[2];
				h_result_2d[idx]+=h_integrand2(h_params_2d+idx*2,h_theta,h_phi)*h_w[idx1]*h_w[idx2]*h_lim1*h_lim2;
				}
			}
		}
	h_end_t=clock();
	h_time=(float)(h_end_t-h_start_t)/CLOCKS_PER_SEC;	

	//Print time taken and up to 20 results
	printf("CPU time elapsed: %f\n",h_time);
	printf("GPU time elapsed: %f\n",d_time);
	printf("           Single Integral            |           Double Integral\n");
	printf("    GPU     |    CPU     |   Exact    |    GPU     |    CPU     |   Exact \n");
	for(idx=0;idx<20;idx++)
		{
		printf("%E %E %E %E %E %E\n",h_result_1d[idx+number_of_integrals_1d],h_result_1d[idx],h_integral1_exact(h_params_1d+idx*2,h_lims_1d),h_result_2d[idx+number_of_integrals_2d],h_result_2d[idx],h_integral2_exact(h_params_2d+idx*2,h_lims_2d));
		}

   //Clean up
   free(h_x);
   free(h_w);
   free(h_params_1d);
   free(h_params_2d);
   free(h_lims_1d);
   free(h_lims_2d);
   free(h_result_1d);
   free(h_result_2d);
   cudaFree(d_x);
   cudaFree(d_w);
   cudaFree(d_params_1d);
   cudaFree(d_params_2d);
   cudaFree(d_lims_1d);
   cudaFree(d_lims_2d);
   cudaFree(d_result);
   cudaDeviceReset();
}
