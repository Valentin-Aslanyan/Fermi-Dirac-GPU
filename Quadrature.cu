#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <lapacke.h>
#include <cblas.h>
#include <lapacke_mangling.h>
#include <lapacke_utils.h>

/*
Here we demonstrate the calculation of single and double integrals on GPUs using CUDA
The prefix h_ corresponds to host (namely, CPU)
The prefix d_ corresponds to device (namely, GPU)
Specifically, we evaluate the integrals:
\int_3^5  A \theta^2 + \exp(B\theta) d\theta			(1)
\int_3^5 \int_2^10  A \phi\exp(B\theta\phi) d\theta d\phi	(2)
A and B are constants, stored in h_params[] and d_params[]
The limits are stored in h_limits[] and d_limits[]
*/


//The following two functions carry out a single integral (1)
__device__ double d_integrand1(double *d_params,double theta)
{
	double integrand=d_params[0]*theta*theta+d_params[1]*exp(theta);
	return integrand;
}

__global__ void d_single_integral(double *d_params, double *d_limits, double *d_result, double *d_w, double *d_x)
{	
	//Calculate the integrand on each thread
	int tid=threadIdx.x;
	extern  __shared__ double temp[];	//Integrand is stored in shared memory
	double *d_result_temp=&temp[0];
	double lim=d_limits[1]-d_limits[0];
	double theta=d_x[tid]*lim+d_limits[0];
	double integrand=d_integrand1(d_params,theta);
	d_result_temp[tid]=integrand*w[tid]*lim;

	//Perform a standard reduction to integrate
    for (int d = blockDim.x>>1; d > 0; d >>= 1)
    {
      __syncthreads(); 
      if (threadIdx.x<d)
	{
		d_result_temp[tid]   += d_result_temp[tid+d];
	}
    }
	__syncthreads();
    if (threadIdx.x==0) 
    {	
	d_result[blockIdx.x]=d_result_temp[tid];
    }
}

//Exact, analytic solution
double h_integral1_exact(double *h_params, double *h_limits)
{
	return h_params[0]*(h_limits[1]*h_limits[1]*h_limits[1]-h_limits[0]*h_limits[0]*h_limits[0])/3.0+h_params[1]*(exp(h_params[1]*h_limits[1])-exp(h_params[1]*h_limits[0]))
}


//Two functions to carry out double integral (2)
__device__ double d_integrand2(double *d_params,double theta,double phi)
{
	double integrand=d_params[0]*phi*exp(d_params[0]*theta*phi);
	return integrand;
}

//Involves now two reductions and the intermediate results stored in main GPU memory
__global__ void d_double_integral(double *d_params, double *d_limits, double *d_result,double *d_result_part, double *d_w, double *d_x)
{	
	//Calculate the integrand on each thread
	int tid=threadIdx.x, d;
	extern  __shared__ double temp[];
	double *d_result_temp=&temp[0];
	double lim1=d_limits[1]-d_limits[0];
	double lim2=d_limits[3]-d_limits[2];
	double theta=x[tid]*lim1+d_limits[0];
	double phi=x[tid]*lim2+d_limits[2];
	double integrand=d_integrand2(d_params,theta,phi);
	d_result_temp[tid]=integrand*w[tid]*w[blockIdx.x]*lim1*lim2;

	//Series of reductions over the first integral
    for (d = blockDim.x>>1; d > 0; d >>= 1)
    {
      __syncthreads(); 
      if (threadIdx.x<d)
	{
		d_result_temp[tid]   += d_result_temp[tid+d];
	}
    }
	__syncthreads();
	
	//The results of the first integrals are stored in main GPU memory
    if (threadIdx.x==0) 
    {	
	d_result_part[blockIdx.x+gridDim.x*blockIdx.y]=d_result_temp[tid];
    }
	tid=threadIdx.x+blockDim.x*blockIdx.y;


	//Reduction over the second integral to obtain a value for the full double integral
    if(blockIdx.x==0)
    {
	d_result_temp[threadIdx.x]=d_result_part[tid];
 	   for (d = blockDim.x>>1; d > 0; d >>= 1)
 	   {
 	     __syncthreads(); 
 	     if (threadIdx.x<d)
		{
			d_result_temp[threadIdx.x]   += d_result_temp[threadIdx.x+d];
		}
	    }
		__syncthreads();
	    if (threadIdx.x==0) 
	    {	
		d_result[blockIdx.y]=d_result_temp[threadIdx.x];
	    }
    }
}


//Set up the nodes (x) and weights (w) of Gaussian quadrature, using the Golub Welsch algorithm
//coeffs and vectors are allocated arrays which are required only for solving the tridiagonal matrix equation
void gauss_integration_setup(int datapoints, double *weights, double *x,double *coeffs, double *vectors)
{  int idx;

   x[0]=0.0;
   for (idx=1;idx<datapoints;idx++)
	{
	x[idx]=0.0;
	coeffs[idx-1]=0.5/sqrt(1.0-1.0/(4.0*idx*idx));
	}	

   //dstev finds the eigenvalues and vectors of a symmetric matrix
   LAPACKE_dstev(LAPACK_ROW_MAJOR,'v', datapoints, x, coeffs, vectors, datapoints);

   for (idx=0;idx<datapoints;idx++)
	{
	x[idx]=0.5*(x[idx]+1.0);	//This leads to nodes in the range (0,1)
	weights[idx]=vectors[idx]*vectors[idx];
	}
}

int main(int argc,const char** argv)
{
	//number of nodes for integration
	int h_datapoints=32;

	//Initialisation of GPU
	int device_count=0;
	cudaGetDeviceCount(&device_count);
	cudaSetDevice(0);	//Run on device 0 by default - can be changed if multiple GPUs etc are present

	//Declare, allocate and calculate nodes and weights for integration
	double *h_x,*h_w,*h_c,*h_v;
	h_x=(double*)malloc(h_datapoints*sizeof(double));		
	h_w=(double*)malloc(h_datapoints*sizeof(double));
	h_c=(double*)malloc((h_datapoints-1)*sizeof(double));	
	h_v=(double*)malloc(h_datapoints*h_datapoints*sizeof(double));
	gauss_integration_setup(h_datapoints,h_w,h_x,h_c,h_v);

	//Copy nodes and weights to GPU
	double *d_x, *d_w;
	cudaMemcpy(d_x,h_x,sizeof(double)*h_datapoints,cudaMemcpyHostToDevice);
	cudaMemcpy(d_w,h_w,sizeof(double)*h_datapoints,cudaMemcpyHostToDevice);

	//Integral-specific constants
	int number_of_integrals=20;
	double *h_params, *d_params, *h_lims1, *h_lims2, *d_lims1, *d_lims2;
	h_params=(double*)malloc(sizeof(double)*number_of_integrals*2);
	h_lims1=(double*)malloc(sizeof(double)*2);
	h_lims2=(double*)malloc(sizeof(double)*4);
	cudaMalloc((void **)d_lims1,sizeof(double)*2);
	cudaMalloc((void **)d_lims2,sizeof(double)*4);

}
