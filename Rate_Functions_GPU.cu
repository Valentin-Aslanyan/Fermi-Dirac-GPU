#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
extern "C" {  
#include "Rate_Functions_CPU.h"
}
#include "Rate_Functions_GPU.h"

#define ACC_J 19.013
#define ACC_K 25.253
#define ACC_L 6503.0
#define ACC_Jf 19.013f
#define ACC_Kf 25.253f
#define ACC_Lf 6503.0f

//The prefix d_ corresponds to device (namely, GPU)
//This file contains GPU equivalents to the fully-CPU rate calculations in Rate_Functions_CPU.c
//Readers are advised to familiarise themselves with functions in the former file

//Constants that do not change between calculations
__constant__ int d_excitations_number,d_ionizations_number, d_datapoints, d_block_mult;
__constant__ double d_T_r;
__constant__ float d_T_r_f;

//Functions to evaluate cross sections and collision strengths on the GPU
__device__ double d_j_int(double E0,double E_j,double *B_vector)
{
	double Eq=E_j/E0;
	double integrand=-log(Eq)*B_vector[0]+B_vector[1]+Eq*(B_vector[2]+Eq*B_vector[3]);
	return integrand;
}

__device__ double d_k_int(double E0,double E1,double E_i,double *C_vector)
{	double E_i_pow=E_i*E_i, E0_pow=E0*E0, E1_pow=E1*E1, E1_prime=E0-E_i-E1, E1_prime_pow=E1_prime*E1_prime;
	double a=0.5*(sqrt(E0_pow+4.0*E_i_pow)-E0);
	double b=a+E_i;
	double integrand=(1.0/((E1+a)*(E1+b))+1.0/((E1_prime+a)*(E1_prime+b)))*C_vector[0];
	integrand+=2.0*C_vector[1]/E0;
	integrand+=2.0*C_vector[2]*(E0-E_i)/E0_pow; E0_pow*=E0;
	integrand+=3.0*C_vector[3]*(E1_pow+E1_prime_pow)/E0_pow; E0_pow*=E0; E1_pow*=E1; E1_prime_pow*=E1_prime;
	integrand+=4.0*C_vector[4]*(E1_pow+E1_prime_pow)/E0_pow;
	integrand*=0.5/E_i;
	return integrand;
}

__device__ double d_l_int(double EGamma,double T_r,double *D_vector)
{	double exp_EG=exp(EGamma/T_r)-1.0;
	double integrand=(D_vector[0]+D_vector[1]/EGamma)/exp_EG;
	return integrand;
}

__device__ float d_j_int_f(float E0,float E_j,float *B_vector)
{
	float Eq=E_j/E0;
	float integrand=-logf(Eq)*B_vector[0]+B_vector[1]+Eq*(B_vector[2]+Eq*B_vector[3]);
	return integrand;
}


__device__ float d_k_int_f(float E0,float E1,float E_i,float *C_vector)
{	float E_i_pow=E_i*E_i, E0_pow=E0*E0, E1_pow=E1*E1, E1_prime=E0-E_i-E1, E1_prime_pow=E1_prime*E1_prime;
	float a=0.5f*(sqrtf(E0_pow+4.0f*E_i_pow)-E0);
	float b=a+E_i;
	float integrand=(1.0f/((E1+a)*(E1+b))+1.0f/((E1_prime+a)*(E1_prime+b)))*C_vector[0];
	integrand+=2.0f*C_vector[1]/E0;
	integrand+=2.0f*C_vector[2]*(E0-E_i)/E0_pow; E0_pow*=E0;
	integrand+=3.0f*C_vector[3]*(E1_pow+E1_prime_pow)/E0_pow; E0_pow*=E0; E1_pow*=E1; E1_prime_pow*=E1_prime;
	integrand+=4.0f*C_vector[4]*(E1_pow+E1_prime_pow)/E0_pow;
	integrand*=0.5f/E_i;
	return integrand;
}

__device__ float d_l_int_f(float EGamma,float T_r,float *D_vector)
{	float exp_EG=expf(EGamma/T_r)-1.0f;
	float integrand=(D_vector[0]+D_vector[1]/EGamma)/exp_EG;
	return integrand;
}

//Carry out an integral for the collisional excitation coefficient
__global__ void d_j_calc(double *d_params, double *E_j, double *B_vector, double *d_j_up, double *w, double *x)
{	
	//Calculate the integrand on each thread
	int node_num, integral_num;
	integral_num=threadIdx.x/d_datapoints;
	node_num=threadIdx.x % d_datapoints;
	integral_num+=blockIdx.x*d_block_mult;
	extern  __shared__ double d_j_up_temp[];	//Integrand is stored in shared memory
	//double *d_j_up_temp=&temp[0];
	double lim=(E_j[integral_num]+2.0*fabs(d_params[1])+ACC_J*d_params[0]);
	double E0=x[node_num]*lim+E_j[integral_num];
	double integrand=d_j_int(E0,E_j[integral_num],B_vector+integral_num*4)*w[node_num];
	double fermi=1.0/(1.0+exp((E0-d_params[1])/d_params[0]));
	double fermi_m=1.0/(1.0+exp((E0-E_j[integral_num]-d_params[1])/d_params[0]));
	d_j_up_temp[threadIdx.x]=integrand*fermi*(1.0-fermi_m);

	//Perform a standard reduction to integrate
    for (int d = d_datapoints>>1; d > 0; d >>= 1)
    {
      __syncthreads(); 
      if (node_num<d)
	{
		d_j_up_temp[threadIdx.x]   += d_j_up_temp[threadIdx.x+d];
	}
    }
	__syncthreads();
    if (node_num==0) 
    {	
	d_j_up[integral_num]=d_j_up_temp[threadIdx.x]*lim;
    }
}

__global__ void d_j_calc_f(float *d_params, float *E_j, float *B_vector, float *d_j_up, float *w, float *x)
{	
	//Calculate the integrand on each thread
	int node_num, integral_num;
	integral_num=threadIdx.x/d_datapoints;
	node_num=threadIdx.x % d_datapoints;
	integral_num+=blockIdx.x*d_block_mult;
	extern  __shared__ float d_j_up_tempf[];	//Integrand is stored in shared memory
	//float *d_j_up_temp=&temp[0];
	float lim=(E_j[integral_num]+2.0f*fabsf(d_params[1])+ACC_Jf*d_params[0]);
	float E0=x[node_num]*lim+E_j[integral_num];
	float integrand=d_j_int_f(E0,E_j[integral_num],B_vector+integral_num*4)*w[node_num];
	float fermi=1.0f/(1.0f+expf((E0-d_params[1])/d_params[0]));
	float fermi_m=1.0f/(1.0f+expf((E0-E_j[integral_num]-d_params[1])/d_params[0]));
	d_j_up_tempf[threadIdx.x]=integrand*fermi*(1.0f-fermi_m);

	//Perform a standard reduction to integrate
    for (int d = d_datapoints>>1; d > 0; d >>= 1)
    {
      __syncthreads(); 
      if (node_num<d)
	{
		d_j_up_tempf[threadIdx.x]   += d_j_up_tempf[threadIdx.x+d];
	}
    }
	__syncthreads();
    if (node_num==0) 
    {	
	d_j_up[integral_num]=d_j_up_tempf[threadIdx.x]*lim;
    }
}

//Carry out a double integral for the collisional ionization coefficient
//Involves now two reductions and the intermediate results stored in main GPU memory
__global__ void d_k_calc(double *d_params, double *E_i, double *C_vector,double *d_k_up, double *w, double *x)
{	
	//Calculate the integrand on each thread
	int node_num, integral_num;
	integral_num=threadIdx.x/d_datapoints;
	node_num=threadIdx.x % d_datapoints;
	integral_num+=blockIdx.x*d_block_mult;
	extern  __shared__ double d_k_up_temp[];
	//double *d_k_up_temp=&temp[0];
	double lim=(fabs(d_params[1])+50.0+ACC_K*d_params[0]);
	double E0prime=x[node_num]*lim;
	double E0=E0prime+E_i[integral_num];
	double fermiE0=1.0/(1.0+exp((E0-d_params[1])/d_params[0]));
	double E1, integrand, fermiE1, fermiE0prime, int_w_w=0;
	for (int idx=0;idx<d_datapoints;idx++)
	{
		E1=x[idx]*E0prime;
		integrand=d_k_int(E0,E1,E_i[integral_num],C_vector+integral_num*5);
		fermiE1=1.0/(1.0+exp((E1-d_params[1])/d_params[0]));
		fermiE0prime=1.0/(1.0+exp((E0prime-E1-d_params[1])/d_params[0]));
		int_w_w+=integrand*w[node_num]*w[idx]*E0prime*fermiE0*(1.0-fermiE1)*(1.0-fermiE0prime);
	}
	d_k_up_temp[threadIdx.x]=int_w_w;

 	//Perform a standard reduction to integrate
    for (int d = d_datapoints>>1; d > 0; d >>= 1)
    {
      __syncthreads(); 
      if (node_num<d)
	{
		d_k_up_temp[threadIdx.x]   += d_k_up_temp[threadIdx.x+d];
	}
    }
	__syncthreads();
    if (node_num==0) 
    {	
	d_k_up[integral_num]=d_k_up_temp[threadIdx.x]*lim;
    }
}

__global__ void d_k_calc_f(float *d_params, float *E_i, float *C_vector,float *d_k_up, float *w, float *x)
{	
	//Calculate the integrand on each thread
	int node_num, integral_num;
	integral_num=threadIdx.x/d_datapoints;
	node_num=threadIdx.x % d_datapoints;
	integral_num+=blockIdx.x*d_block_mult;
	extern  __shared__ float d_k_up_tempf[];
	//float *d_k_up_temp=&temp[0];
	float lim=(fabsf(d_params[1])+50.0f+ACC_Kf*d_params[0]);
	float E0prime=x[node_num]*lim;
	float E0=E0prime+E_i[integral_num];
	float fermiE0=1.0f/(1.0f+exp((E0-d_params[1])/d_params[0]));
	float E1, integrand, fermiE1, fermiE0prime, int_w_w=0.0f;
	for (int idx=0;idx<d_datapoints;idx++)
	{
		E1=x[idx]*E0prime;
		integrand=d_k_int_f(E0,E1,E_i[integral_num],C_vector+integral_num*5);
		fermiE1=1.0f/(1.0f+exp((E1-d_params[1])/d_params[0]));
		fermiE0prime=1.0f/(1.0f+exp((E0prime-E1-d_params[1])/d_params[0]));
		int_w_w+=integrand*w[node_num]*w[idx]*E0prime*fermiE0*(1.0f-fermiE1)*(1.0f-fermiE0prime);
	}
	d_k_up_tempf[threadIdx.x]=int_w_w;

	//Series of reductions over the first integral
    for (int d = d_datapoints>>1; d > 0; d >>= 1)
    {
      __syncthreads(); 
      if (node_num<d)
	{
		d_k_up_tempf[threadIdx.x]   += d_k_up_tempf[threadIdx.x+d];
	}
    }
	__syncthreads();
    if (node_num==0) 
    {	
	d_k_up[integral_num]=d_k_up_tempf[threadIdx.x]*lim;
    }
}

//Carry out an integral for the photoionization coefficient and energy change due to photoionization
//Single integral, similar to d_j_calc()
__global__ void d_l_calc(double *d_params, double *E_i, double *D_vector, double *d_l, double *d_le, double *w, double *x)
{	
	int node_num, integral_num;
	integral_num=threadIdx.x/d_datapoints;
	node_num=threadIdx.x % d_datapoints;
	integral_num+=blockIdx.x*d_block_mult;
	extern  __shared__ double d_l_temp[];
	//double *d_l_temp=&temp[0];
	double *d_le_temp=&d_l_temp[blockDim.x];
	double lim=ACC_L;
	if(d_params[1]>0.0){lim+=d_params[1];}
	double EGammaPrime=x[node_num]*lim;
	double EGamma=EGammaPrime+E_i[integral_num];
	double fermi_m=1.0-1.0/(1.0+exp((EGammaPrime-d_params[1])/d_params[0]));
	double integrand=d_l_int(EGamma,d_T_r,D_vector+integral_num*2)*w[node_num]*fermi_m;
	d_l_temp[threadIdx.x]=integrand;
	d_le_temp[threadIdx.x]=integrand*EGammaPrime;
    for (int d = d_datapoints>>1; d > 0; d >>= 1)
    {
      __syncthreads(); 
      if (node_num<d)
	{
		d_l_temp[threadIdx.x]   += d_l_temp[threadIdx.x+d];
		d_le_temp[threadIdx.x]   += d_le_temp[threadIdx.x+d];
	}
    }
	__syncthreads();
    if (node_num==0) 
    {	
	d_l[integral_num]=d_l_temp[threadIdx.x]*lim;
	d_le[integral_num]=d_le_temp[threadIdx.x]*lim;
    }
}

__global__ void d_l_calc_f(float *d_params, float *E_i, float *D_vector, float *d_l, float *d_le, float *w, float *x)
{	
	int node_num, integral_num;
	integral_num=threadIdx.x/d_datapoints;
	node_num=threadIdx.x % d_datapoints;
	integral_num+=blockIdx.x*d_block_mult;
	extern  __shared__ float d_l_tempf[];
	//float *d_l_temp=&temp[0];
	float *d_le_temp=&d_l_tempf[blockDim.x];
	float lim=ACC_Lf;
	if(d_params[1]>0.0f){lim+=d_params[1];}
	float EGammaPrime=x[node_num]*lim;
	float EGamma=EGammaPrime+E_i[integral_num];
	float fermi_m=1.0f-1.0f/(1.0f+expf((EGammaPrime-d_params[1])/d_params[0]));
	float integrand=d_l_int_f(EGamma,d_T_r_f,D_vector+integral_num*2)*w[node_num]*fermi_m;
	d_l_tempf[threadIdx.x]=integrand;
	d_le_temp[threadIdx.x]=integrand*EGammaPrime;

    for (int d = d_datapoints>>1; d > 0; d >>= 1)
    {
      __syncthreads(); 
      if (node_num<d)
	{
		d_l_tempf[threadIdx.x]   += d_l_tempf[threadIdx.x+d];
		d_le_temp[threadIdx.x]   += d_le_temp[threadIdx.x+d];
	}
    }
	__syncthreads();
    if (node_num==0) 
    {	
	d_l[integral_num]=d_l_tempf[threadIdx.x]*lim;
	d_le[integral_num]=d_le_temp[threadIdx.x]*lim;
    }
}

void d_setup(double **d_params, double **d_B_vector, double **d_C_vector, double **d_D_vector, double **d_E_j, double **d_E_i, double **d_j, double **d_k, double **d_l, double **d_x, double **d_w, double *B_vector, double *C_vector, double *D_vector, double *E_j, double *E_i, double T_r, double *h_x, double *h_w, int ionizations_number, int excitations_number, int h_datapoints, cudaStream_t *streams, int h_block_mult)
{
cudaMalloc((void **)d_params,sizeof(double)*2);
cudaMalloc((void **)d_B_vector,sizeof(double)*excitations_number*4);
cudaMalloc((void **)d_C_vector,sizeof(double)*ionizations_number*5);
cudaMalloc((void **)d_D_vector,sizeof(double)*ionizations_number*2);
cudaMalloc((void **)d_E_j,sizeof(double)*excitations_number);
cudaMalloc((void **)d_E_i,sizeof(double)*ionizations_number);
cudaMalloc((void **)d_j,sizeof(double)*excitations_number);
cudaMalloc((void **)d_k,sizeof(double)*ionizations_number);
cudaMalloc((void **)d_l,2*sizeof(double)*ionizations_number);
cudaMalloc((void **)d_x,sizeof(double)*h_datapoints);
cudaMalloc((void **)d_w,sizeof(double)*h_datapoints);

cudaMemcpyToSymbol(d_ionizations_number,&ionizations_number,sizeof(ionizations_number));
cudaMemcpyToSymbol(d_excitations_number,&excitations_number,sizeof(excitations_number));
cudaMemcpyToSymbol(d_datapoints,&h_datapoints,sizeof(h_datapoints));
cudaMemcpyToSymbol(d_block_mult,&h_block_mult,sizeof(h_block_mult));
cudaMemcpyToSymbol(d_T_r,&T_r,sizeof(T_r));
cudaMemcpy(*d_B_vector,B_vector,sizeof(double)*excitations_number*4,cudaMemcpyHostToDevice);
cudaMemcpy(*d_C_vector,C_vector,sizeof(double)*ionizations_number*5,cudaMemcpyHostToDevice);
cudaMemcpy(*d_D_vector,D_vector,sizeof(double)*ionizations_number*2,cudaMemcpyHostToDevice);
cudaMemcpy(*d_E_j,E_j,sizeof(double)*excitations_number,cudaMemcpyHostToDevice);
cudaMemcpy(*d_E_i,E_i,sizeof(double)*ionizations_number,cudaMemcpyHostToDevice);
cudaMemcpy(*d_x,h_x,sizeof(double)*h_datapoints,cudaMemcpyHostToDevice);
cudaMemcpy(*d_w,h_w,sizeof(double)*h_datapoints,cudaMemcpyHostToDevice);

cudaStreamCreate(&streams[0]);
cudaStreamCreate(&streams[1]);
}

void d_setup_f(float **d_params, float **d_B_vector, float **d_C_vector, float **d_D_vector, float **d_E_j, float **d_E_i, float **d_j, float **d_k, float **d_l, float **d_x, float **d_w, float *B_vector, float *C_vector, float *D_vector, float *E_j, float *E_i, float T_r, float *h_x, float *h_w, int ionizations_number, int excitations_number, int h_datapoints, cudaStream_t *streams, int h_block_mult)
{
cudaMalloc((void **)d_params,sizeof(float)*2);
cudaMalloc((void **)d_B_vector,sizeof(float)*excitations_number*4);
cudaMalloc((void **)d_C_vector,sizeof(float)*ionizations_number*5);
cudaMalloc((void **)d_D_vector,sizeof(float)*ionizations_number*2);
cudaMalloc((void **)d_E_j,sizeof(float)*excitations_number);
cudaMalloc((void **)d_E_i,sizeof(float)*ionizations_number);
cudaMalloc((void **)d_j,sizeof(float)*excitations_number);
cudaMalloc((void **)d_k,sizeof(float)*ionizations_number);
cudaMalloc((void **)d_l,2*sizeof(float)*ionizations_number);
cudaMalloc((void **)d_x,sizeof(float)*h_datapoints);
cudaMalloc((void **)d_w,sizeof(float)*h_datapoints);

cudaMemcpyToSymbol(d_ionizations_number,&ionizations_number,sizeof(ionizations_number));
cudaMemcpyToSymbol(d_excitations_number,&excitations_number,sizeof(excitations_number));
cudaMemcpyToSymbol(d_datapoints,&h_datapoints,sizeof(h_datapoints));
cudaMemcpyToSymbol(d_block_mult,&h_block_mult,sizeof(h_block_mult));
cudaMemcpyToSymbol(d_T_r_f,&T_r,sizeof(T_r));
cudaMemcpy(*d_B_vector,B_vector,sizeof(float)*excitations_number*4,cudaMemcpyHostToDevice);
cudaMemcpy(*d_C_vector,C_vector,sizeof(float)*ionizations_number*5,cudaMemcpyHostToDevice);
cudaMemcpy(*d_D_vector,D_vector,sizeof(float)*ionizations_number*2,cudaMemcpyHostToDevice);
cudaMemcpy(*d_E_j,E_j,sizeof(float)*excitations_number,cudaMemcpyHostToDevice);
cudaMemcpy(*d_E_i,E_i,sizeof(float)*ionizations_number,cudaMemcpyHostToDevice);
cudaMemcpy(*d_x,h_x,sizeof(float)*h_datapoints,cudaMemcpyHostToDevice);
cudaMemcpy(*d_w,h_w,sizeof(float)*h_datapoints,cudaMemcpyHostToDevice);

cudaStreamCreate(&streams[0]);
cudaStreamCreate(&streams[1]);
}

void d_cleanup(double *d_params, double *d_B_vector, double *d_C_vector, double *d_E_j, double *d_E_i, double *d_j, double *d_k, double *d_l, double *d_x, double *d_w)
{
   cudaFree(d_params);
   cudaFree(d_B_vector);
   cudaFree(d_C_vector);
   cudaFree(d_E_j);
   cudaFree(d_E_i);
   cudaFree(d_j);
   cudaFree(d_k);
   cudaFree(d_l);
   cudaFree(d_x);
   cudaFree(d_w);
   cudaDeviceReset();
}

void d_cleanup_f(float *d_params, float *d_B_vector, float *d_C_vector, float *d_E_j, float *d_E_i, float *d_j, float *d_k, float *d_l, float *d_x, float *d_w)
{
   cudaFree(d_params);
   cudaFree(d_B_vector);
   cudaFree(d_C_vector);
   cudaFree(d_E_j);
   cudaFree(d_E_i);
   cudaFree(d_j);
   cudaFree(d_k);
   cudaFree(d_l);
   cudaFree(d_x);
   cudaFree(d_w);
   cudaDeviceReset();
}

void d_calculate_rates(double *d_params,double *d_B_vector, double *d_C_vector, double *d_D_vector, double *d_E_j, double *d_E_i, double *d_j, double *d_k, double *d_l, double *d_x, double *d_w,double *h_params, double *h_j,double *h_k,double *h_l,double *h_w, double *h_x, double  *ib_E, double n_e, double T_r, double *charge_vector, double *N, int states_number, int ionizations_number,int excitations_number,int h_datapoints,cudaStream_t *streams, int h_block_mult)
{	
	cudaMemcpy(d_params,h_params,sizeof(double)*2,cudaMemcpyHostToDevice);
	d_j_calc<<<excitations_number/h_block_mult,h_datapoints*h_block_mult,h_datapoints*h_block_mult*sizeof(double),streams[0]>>>(d_params,d_E_j,d_B_vector,d_j,d_w,d_x);
	d_l_calc<<<ionizations_number/h_block_mult,h_datapoints*h_block_mult,2*h_datapoints*h_block_mult*sizeof(double),streams[0]>>>(d_params,d_E_i,d_D_vector,d_l,d_l+ionizations_number,d_w,d_x);
	d_k_calc<<<ionizations_number/h_block_mult,h_datapoints*h_block_mult,h_datapoints*h_block_mult*sizeof(double),streams[1]>>>(d_params,d_E_i,d_C_vector,d_k,d_w,d_x);
	*ib_E=h_ib_gauss_integration(h_params[0],n_e,h_params[1],T_r,states_number,charge_vector,N,h_datapoints,h_w,h_x);
	cudaMemcpyAsync(h_j,d_j,sizeof(double)*excitations_number,cudaMemcpyDeviceToHost,streams[0]);
	cudaMemcpyAsync(h_l,d_l,2*sizeof(double)*ionizations_number,cudaMemcpyDeviceToHost,streams[0]);
	cudaMemcpyAsync(h_k,d_k,sizeof(double)*ionizations_number,cudaMemcpyDeviceToHost,streams[1]);
	cudaDeviceSynchronize();
}

void d_calculate_rates_f(float *d_params,float *d_B_vector, float *d_C_vector, float *d_D_vector, float *d_E_j, float *d_E_i, float *d_j, float *d_k, float *d_l, float *d_x, float *d_w,float *h_params, float *h_j,float *h_k,float *h_l,float *h_w, float *h_x, float  *ib_E, float n_e, float T_r, float *charge_vector, float *N, int states_number, int ionizations_number,int excitations_number,int h_datapoints,cudaStream_t *streams, int h_block_mult)
{	
	cudaMemcpy(d_params,h_params,sizeof(float)*2,cudaMemcpyHostToDevice);
	d_j_calc_f<<<excitations_number/h_block_mult,h_datapoints*h_block_mult,h_datapoints*h_block_mult*sizeof(float),streams[0]>>>(d_params,d_E_j,d_B_vector,d_j,d_w,d_x);
	d_l_calc_f<<<ionizations_number/h_block_mult,h_datapoints*h_block_mult,2*h_datapoints*h_block_mult*sizeof(float),streams[0]>>>(d_params,d_E_i,d_D_vector,d_l,d_l+ionizations_number,d_w,d_x);
	d_k_calc_f<<<ionizations_number/h_block_mult,h_datapoints*h_block_mult,h_datapoints*h_block_mult*sizeof(float),streams[1]>>>(d_params,d_E_i,d_C_vector,d_k,d_w,d_x);
	*ib_E=h_ib_gauss_integration_f(h_params[0],n_e,h_params[1],T_r,states_number,charge_vector,N,h_datapoints,h_w,h_x);
	cudaMemcpyAsync(h_j,d_j,sizeof(float)*excitations_number,cudaMemcpyDeviceToHost,streams[0]);
	cudaMemcpyAsync(h_l,d_l,2*sizeof(float)*ionizations_number,cudaMemcpyDeviceToHost,streams[0]);
	cudaMemcpyAsync(h_k,d_k,sizeof(float)*ionizations_number,cudaMemcpyDeviceToHost,streams[1]);
	cudaDeviceSynchronize();
}

//This is the RK-4 solver using GPUs to calculate atomic rates
//See h_solve_RK4 in Rate_Functions_CPU.c for more details
void hd_solve_RK4(int states_number, int ionizations_number, int excitations_number, double delta, double *charge_vector, double *N, double *N_temp1, double *N_temp2, double *N_temp3, double *N_temp4, double *IntE_temp, double *n_e, double *T_e, double *T_F, double *Internal_Energy,int h_datapoints, double *h_w, double *h_x, double *h_j, double *h_k, double *h_l, double T_r, int *excitations_indices, int *ionizations_indices, double *E_i, double *E_j,double *A_vector, double *B_vector, double *C_vector, double *D_vector, double *R_1, double *R_2, double *h_params, double *d_params, double *d_B_vector, double *d_C_vector, double *d_D_vector, double *d_E_j, double *d_E_i, double *d_j, double *d_k, double *d_l, double *d_x, double *d_w,cudaStream_t *streams, int h_block_mult)
{	//RK-4 method, N_t[n+1] = N_t[n]+Delta_t/6*(k_1+2k_2+2k_3+k_4)
	//Declare variables, copy current values to temporary arrays
	double n_e_temp, T_F_temp, T_e_temp, delta_2=0.5*delta, Int_energy_temp1, Int_energy_temp2, Int_energy_temp3, Int_energy_temp4, ib_E;
	cblas_dcopy(states_number,N,1,N_temp1,1);
	cblas_dcopy(states_number,N,1,N_temp2,1);
	cblas_dcopy(states_number,N,1,N_temp3,1);
	cblas_dcopy(states_number,N,1,N_temp4,1);

	//Coefficients 1
	h_params[0]=*T_e;
	h_params[1]=Get_Chemical_Potential(*T_e,*T_F);
	d_calculate_rates(d_params,d_B_vector, d_C_vector,d_D_vector, d_E_j, d_E_i, d_j, d_k, d_l, d_x,  d_w,h_params,h_j,h_k,h_l,h_w,h_x,&ib_E,*n_e,T_r,charge_vector,N, states_number, ionizations_number,excitations_number,h_datapoints,streams,h_block_mult);
	h_create_rate_matrices(states_number,ionizations_number,excitations_number,*T_e,*n_e,h_params[1] ,R_1,R_2,h_j,h_k, h_l,excitations_indices, ionizations_indices,E_i,E_j);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_1,states_number,N,1,1.0,N_temp1,1);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_2,states_number,N,1,0.0,IntE_temp,1);
	Int_energy_temp1=vector_sum(states_number,IntE_temp)+*Internal_Energy+ib_E*delta_2;

	//Coefficients 2
	n_e_temp=Get_n_e(states_number,N_temp1,charge_vector);
	T_F_temp=Fermi_Energy(n_e_temp);
	T_e_temp=T_F_temp*Invert_C_V(Int_energy_temp1/(T_F_temp*n_e_temp));
	h_params[0]=T_e_temp;
	h_params[1]=Get_Chemical_Potential(T_e_temp,T_F_temp);
	d_calculate_rates(d_params,d_B_vector, d_C_vector,d_D_vector, d_E_j, d_E_i, d_j, d_k, d_l, d_x,  d_w,h_params,h_j,h_k,h_l,h_w,h_x,&ib_E,n_e_temp,T_r,charge_vector,N_temp1, states_number, ionizations_number,excitations_number,h_datapoints,streams,h_block_mult);
	h_create_rate_matrices(states_number,ionizations_number,excitations_number,T_e_temp,n_e_temp,h_params[1],R_1,R_2,h_j,h_k,h_l,excitations_indices,ionizations_indices,E_i,E_j);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_1,states_number,N_temp1,1,1.0,N_temp2,1);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_2,states_number,N_temp1,1,0.0,IntE_temp,1);
	Int_energy_temp2=vector_sum(states_number,IntE_temp)+*Internal_Energy+ib_E*delta_2;

	//Coefficients 3
	n_e_temp=Get_n_e(states_number,N_temp2,charge_vector);
	T_F_temp=Fermi_Energy(n_e_temp);
	T_e_temp=T_F_temp*Invert_C_V(Int_energy_temp2/(T_F_temp*n_e_temp));
	h_params[0]=T_e_temp;
	h_params[1]=Get_Chemical_Potential(T_e_temp,T_F_temp);
	d_calculate_rates(d_params,d_B_vector, d_C_vector,d_D_vector, d_E_j, d_E_i, d_j, d_k, d_l, d_x,  d_w,h_params,h_j,h_k,h_l,h_w,h_x,&ib_E,n_e_temp,T_r,charge_vector,N_temp2, states_number, ionizations_number,excitations_number,h_datapoints,streams,h_block_mult);
	h_create_rate_matrices(states_number,ionizations_number,excitations_number,T_e_temp,n_e_temp,h_params[1],R_1,R_2,h_j,h_k,h_l,excitations_indices,ionizations_indices,E_i,E_j);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta,R_1,states_number,N_temp2,1,1.0,N_temp3,1);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta,R_2,states_number,N_temp2,1,0.0,IntE_temp,1);
	Int_energy_temp3=vector_sum(states_number,IntE_temp)+*Internal_Energy+ib_E*delta;

	//Coefficients 4
	n_e_temp=Get_n_e(states_number,N_temp3,charge_vector);
	T_F_temp=Fermi_Energy(n_e_temp);
	T_e_temp=T_F_temp*Invert_C_V(Int_energy_temp3/(T_F_temp*n_e_temp));
	h_params[0]=T_e_temp;
	h_params[1]=Get_Chemical_Potential(T_e_temp,T_F_temp);
	d_calculate_rates(d_params,d_B_vector, d_C_vector,d_D_vector, d_E_j, d_E_i, d_j, d_k, d_l, d_x,  d_w,h_params,h_j,h_k,h_l,h_w,h_x,&ib_E,n_e_temp,T_r,charge_vector,N_temp3, states_number, ionizations_number,excitations_number,h_datapoints,streams,h_block_mult);
	h_create_rate_matrices(states_number,ionizations_number,excitations_number,T_e_temp,n_e_temp,h_params[1],R_1,R_2,h_j,h_k,h_l,excitations_indices,ionizations_indices,E_i,E_j);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_1,states_number,N_temp3,1,-1.0,N_temp4,1);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_2,states_number,N_temp2,1,0.0,IntE_temp,1);
	Int_energy_temp4=vector_sum(states_number,IntE_temp)-*Internal_Energy+ib_E*delta_2;

	//Calculate starting values for next iteration
	cblas_daxpy(states_number,1.0,N_temp1,1,N_temp3,1);
	cblas_daxpy(states_number,2.0,N_temp2,1,N_temp4,1);
	cblas_daxpy(states_number,1.0,N_temp3,1,N_temp4,1);
 	cblas_dscal(states_number,0.3333333333333333,N_temp4,1);
	cblas_dcopy(states_number,N_temp4,1,N,1);
	*Internal_Energy=(Int_energy_temp1+2.0*Int_energy_temp2+Int_energy_temp3+Int_energy_temp4)*0.3333333333333333;
	*n_e=Get_n_e(states_number,N,charge_vector);
	*T_F=Fermi_Energy(*n_e);
	*T_e=*T_F*Invert_C_V(*Internal_Energy/(*T_F* *n_e));
}

void hd_solve_RK4_f(int states_number, int ionizations_number, int excitations_number, float delta, float *charge_vector, float *N, float *N_temp1, float *N_temp2, float *N_temp3, float *N_temp4, float *IntE_temp, float *n_e, float *T_e, float *T_F, float *Internal_Energy,int h_datapoints, float *h_w, float *h_x, float *h_j, float *h_k, float *h_l, float T_r, int *excitations_indices, int *ionizations_indices, float *E_i, float *E_j,float *A_vector, float *B_vector, float *C_vector, float *D_vector, float *R_1, float *R_2, float *h_params, float *d_params, float *d_B_vector, float *d_C_vector, float *d_D_vector, float *d_E_j, float *d_E_i, float *d_j, float *d_k, float *d_l, float *d_x, float *d_w,cudaStream_t *streams, int h_block_mult)
{	//RK-4 method, N_t[n+1] = N_t[n]+Delta_t/6*(k_1+2k_2+2k_3+k_4)
	//Declare variables, copy current values to temporary arrays
	float n_e_temp, T_F_temp, T_e_temp, delta_2=0.5f*delta, Int_energy_temp1, Int_energy_temp2, Int_energy_temp3, Int_energy_temp4, ib_E;
	cblas_scopy(states_number,N,1,N_temp1,1);
	cblas_scopy(states_number,N,1,N_temp2,1);
	cblas_scopy(states_number,N,1,N_temp3,1);
	cblas_scopy(states_number,N,1,N_temp4,1);

	//Coefficients 1
	h_params[0]=*T_e;
	h_params[1]=Get_Chemical_Potential(*T_e,*T_F);
	d_calculate_rates_f(d_params,d_B_vector, d_C_vector,d_D_vector, d_E_j, d_E_i, d_j, d_k, d_l, d_x,  d_w,h_params,h_j,h_k,h_l,h_w,h_x,&ib_E,*n_e,T_r,charge_vector,N, states_number, ionizations_number,excitations_number,h_datapoints,streams,h_block_mult);
	h_create_rate_matrices_f(states_number,ionizations_number,excitations_number,*T_e,*n_e,h_params[1] ,R_1,R_2,h_j,h_k, h_l,excitations_indices, ionizations_indices,E_i,E_j);
	cblas_sgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_1,states_number,N,1,1.0f,N_temp1,1);
	cblas_sgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_2,states_number,N,1,0.0f,IntE_temp,1);
	Int_energy_temp1=vector_sum_f(states_number,IntE_temp)+*Internal_Energy+ib_E*delta_2;

	//Coefficients 2
	n_e_temp=Get_n_e_f(states_number,N_temp1,charge_vector);
	T_F_temp=Fermi_Energy_f(n_e_temp);
	T_e_temp=T_F_temp*Invert_C_V_f(Int_energy_temp1/(T_F_temp*n_e_temp));
	h_params[0]=T_e_temp;
	h_params[1]=Get_Chemical_Potential_f(T_e_temp,T_F_temp);
	d_calculate_rates_f(d_params,d_B_vector, d_C_vector,d_D_vector, d_E_j, d_E_i, d_j, d_k, d_l, d_x,  d_w,h_params,h_j,h_k,h_l,h_w,h_x,&ib_E,n_e_temp,T_r,charge_vector,N_temp1, states_number, ionizations_number,excitations_number,h_datapoints,streams,h_block_mult);
	h_create_rate_matrices_f(states_number,ionizations_number,excitations_number,T_e_temp,n_e_temp,h_params[1],R_1,R_2,h_j,h_k,h_l,excitations_indices,ionizations_indices,E_i,E_j);
	cblas_sgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_1,states_number,N_temp1,1,1.0f,N_temp2,1);
	cblas_sgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_2,states_number,N_temp1,1,0.0f,IntE_temp,1);
	Int_energy_temp2=vector_sum_f(states_number,IntE_temp)+*Internal_Energy+ib_E*delta_2;

	//Coefficients 3
	n_e_temp=Get_n_e_f(states_number,N_temp2,charge_vector);
	T_F_temp=Fermi_Energy_f(n_e_temp);
	T_e_temp=T_F_temp*Invert_C_V_f(Int_energy_temp2/(T_F_temp*n_e_temp));
	h_params[0]=T_e_temp;
	h_params[1]=Get_Chemical_Potential_f(T_e_temp,T_F_temp);
	d_calculate_rates_f(d_params,d_B_vector, d_C_vector,d_D_vector, d_E_j, d_E_i, d_j, d_k, d_l, d_x,  d_w,h_params,h_j,h_k,h_l,h_w,h_x,&ib_E,n_e_temp,T_r,charge_vector,N_temp2, states_number, ionizations_number,excitations_number,h_datapoints,streams,h_block_mult);
	h_create_rate_matrices_f(states_number,ionizations_number,excitations_number,T_e_temp,n_e_temp,h_params[1],R_1,R_2,h_j,h_k,h_l,excitations_indices,ionizations_indices,E_i,E_j);
	cblas_sgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta,R_1,states_number,N_temp2,1,1.0f,N_temp3,1);
	cblas_sgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta,R_2,states_number,N_temp2,1,0.0f,IntE_temp,1);
	Int_energy_temp3=vector_sum_f(states_number,IntE_temp)+*Internal_Energy+ib_E*delta;

	//Coefficients 4
	n_e_temp=Get_n_e_f(states_number,N_temp3,charge_vector);
	T_F_temp=Fermi_Energy_f(n_e_temp);
	T_e_temp=T_F_temp*Invert_C_V_f(Int_energy_temp3/(T_F_temp*n_e_temp));
	h_params[0]=T_e_temp;
	h_params[1]=Get_Chemical_Potential_f(T_e_temp,T_F_temp);
	d_calculate_rates_f(d_params,d_B_vector, d_C_vector,d_D_vector, d_E_j, d_E_i, d_j, d_k, d_l, d_x,  d_w,h_params,h_j,h_k,h_l,h_w,h_x,&ib_E,n_e_temp,T_r,charge_vector,N_temp3, states_number, ionizations_number,excitations_number,h_datapoints,streams,h_block_mult);
	h_create_rate_matrices_f(states_number,ionizations_number,excitations_number,T_e_temp,n_e_temp,h_params[1],R_1,R_2,h_j,h_k,h_l,excitations_indices,ionizations_indices,E_i,E_j);
	cblas_sgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_1,states_number,N_temp3,1,-1.0f,N_temp4,1);
	cblas_sgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_2,states_number,N_temp2,1,0.0f,IntE_temp,1);
	Int_energy_temp4=vector_sum_f(states_number,IntE_temp)-*Internal_Energy+ib_E*delta_2;

	//Calculate starting values for next iteration
	cblas_saxpy(states_number,1.0f,N_temp1,1,N_temp3,1);
	cblas_saxpy(states_number,2.0f,N_temp2,1,N_temp4,1);
	cblas_saxpy(states_number,1.0f,N_temp3,1,N_temp4,1);
 	cblas_sscal(states_number,0.3333333333333333f,N_temp4,1);
	cblas_scopy(states_number,N_temp4,1,N,1);
	*Internal_Energy=(Int_energy_temp1+2.0f*Int_energy_temp2+Int_energy_temp3+Int_energy_temp4)*0.3333333333333333f;
	*n_e=Get_n_e_f(states_number,N,charge_vector);
	*T_F=Fermi_Energy_f(*n_e);
	*T_e=*T_F*Invert_C_V_f(*Internal_Energy/(*T_F* *n_e));
}
