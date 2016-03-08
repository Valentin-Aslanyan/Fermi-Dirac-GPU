#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <omp.h>
#include <lapacke.h>
#include <cblas.h>
#include <lapacke_mangling.h>
#include <lapacke_utils.h>

#ifndef ACC_P
#define ACC_P 5
#endif
extern "C" {  
#include "Rate_Functions_CPU.h"
}
#include "Rate_Functions_GPU.h"

__constant__ int d_excitations_number,d_ionizations_number, d_datapoints;
__constant__ double d_T_r;

int main(int argc,const char** argv)
{

//User-specified variables to be changed to modify the calculation
int h_datapoints=32;
double N_T=9E22; 		//cm^-3
double max_time=5E-13;		//s
double delta_time=1E-19;	//s
double T_r=300.0;
int output_frequency=10000;

//CPU variables, memory allocation and construction of the atomic model, non user-specified
//For details, see comments in Rate_Functions_CPU.c
FILE *OUTPUTFILE;
int idx, idx_t, t_iterations=(int)ceil(max_time/delta_time);
double *n_e, *T_e; double T_F, Internal_Energy;
n_e=(double*)malloc((t_iterations+1)*sizeof(double));
T_e=(double*)malloc((t_iterations+1)*sizeof(double));
int states_number=15, ionizations_number=14, excitations_number=14;
double *h_params, *E_i, *E_j, *A_vector, *B_vector1, *B_vector2, *C_vector, *D_vector;
double *h_j, *h_k, *h_l, *h_x, *h_w,*h_c, *h_v;
double *N, *N_temp1, *N_temp2, *N_temp3, *N_temp4, *IntE_temp, *R_1, *R_2, *charge_vector;
int *excitations_indices, *ionizations_indices;
h_allocate_arrays(states_number,ionizations_number,excitations_number,h_datapoints,&h_params,&charge_vector,&E_i,&E_j,&A_vector,&B_vector1,&B_vector2,&C_vector, &D_vector,&h_j, &h_k,&h_l,&h_w,&h_x,&h_c,&h_v,&N,&N_temp1,&N_temp2,&N_temp3,&N_temp4,&IntE_temp,&R_1,&R_2,&excitations_indices,&ionizations_indices);
h_setup_atomic_model(ionizations_number,excitations_number,excitations_indices, ionizations_indices,charge_vector,E_i,E_j,A_vector,B_vector1,C_vector,D_vector);
gauss_integration_setup(h_datapoints,h_w,h_x,h_c,h_v);

//Setup initial conditions - all atoms in ground state and temperature of 1 eV (for stability)
for(idx=0;idx<states_number;idx++){N[idx]=0.0;}
N[0]=N_T;
n_e[0]=Get_n_e(states_number,N,charge_vector);
T_e[0]=1.0;
T_F=Fermi_Energy(n_e[0]);
Internal_Energy=1.5*T_e[0]*(T_e[0]/T_F)*sqrt(T_e[0]/T_F)*Get_C_V(Get_Chemical_Potential(T_e[0],T_F)/T_e[0])*n_e[0];

//GPU variables, memory allocation and the transfer of model-specific constants
//For details, see comments in Rate_Functions_GPU.cu
int device_count=0;
cudaGetDeviceCount(&device_count);
cudaSetDevice(0);	//Run on device 0 by default - can be changed if multiple GPUs etc are present
cudaStream_t streams[2];
double *d_params, *d_B_vector1, *d_C_vector, *d_D_vector, *d_E_j, *d_E_i, *d_j, *d_k, *d_k_part, *d_l, *d_x, *d_w;
d_setup(&d_params,&d_B_vector1,&d_C_vector,&d_D_vector,&d_E_j,&d_E_i, &d_j, &d_k,  &d_k_part, &d_l, &d_x, &d_w,B_vector1, C_vector, D_vector, E_j, E_i, T_r, h_x,h_w, ionizations_number,excitations_number, h_datapoints,streams);

//Open output file
if ((OUTPUTFILE=fopen("CRM_GPU_output.txt", "w"))==NULL)
        {
        printf("Cannot open file! Error!\n");
        exit(2);
       	}

//Run main RK-4 loop
for (idx_t=1;idx_t<=t_iterations;idx_t++)
  {
	//The current values of T_e, n_e are used by the RK-4 calculation and overwritten
	T_e[idx_t]=T_e[idx_t-1];
	n_e[idx_t]=n_e[idx_t-1];
	hd_solve_RK4(states_number, ionizations_number, excitations_number, delta_time, charge_vector, N, N_temp1, N_temp2, N_temp3, N_temp4, IntE_temp, n_e+idx_t, T_e+idx_t, &T_F, &Internal_Energy,h_datapoints,h_w,h_x, h_j, h_k, h_l, T_r, excitations_indices, ionizations_indices,E_i,E_j,A_vector, B_vector1, C_vector, D_vector, R_1, R_2,h_params, d_params, d_B_vector1,d_C_vector, d_D_vector, d_E_j, d_E_i, d_j, d_k, d_k_part, d_l, d_x, d_w,streams);
  }

//Output n_e, T_e
for (idx_t=0;idx_t<=t_iterations;idx_t+=output_frequency)
	{
	fprintf(OUTPUTFILE,"%E %E %E\n",idx_t*delta_time,T_e[idx_t],n_e[idx_t]);
	}

//Clean up memory
d_cleanup(d_params,d_B_vector1, d_C_vector, d_E_j, d_E_i, d_j, d_k, d_k_part, d_l, d_x, d_w);
h_cleanup(h_params, charge_vector, E_i, E_j,A_vector,B_vector, C_vector, D_vector, h_j, h_k, h_l, h_w, h_x, h_c, h_v, N, N_temp1, N_temp2, N_temp3, N_temp4, IntE_temp,  R_1, R_2, excitations_indices, ionizations_indices);

exit(0);
}
