#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <lapacke.h>
#include <cblas.h>
#include <lapacke_mangling.h>
#include <lapacke_utils.h>

#ifndef ACC_P
#define ACC_P 5
#endif
#include "Rate_Functions_CPU.h"

int main(int argc,const char** argv)
{

int h_datapoints=32;
double N_T=9E22; 		//cm^-3
double max_time=5E-13;		//s
double delta_time=1E-19	;	//s
double T_r=300.0;
int output_frequency=10000;

//Declare and allocate CPU variables and atomic model
FILE *OUTPUTFILE;
int idx, idx_t, t_iterations=(int)ceil(max_time/delta_time);
double *n_e, *T_e; double Internal_Energy;
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

//Setup initial conditions
for(idx=0;idx<states_number;idx++){N[idx]=0.0;}
N[0]=N_T;
n_e[0]=Get_n_e(states_number,N,charge_vector);
T_e[0]=1.0;
Internal_Energy=1.5*T_e[0]*n_e[0];

printf("%E %E %E %E\n",n_e[0],T_e[0],0.6666666666666667*Internal_Energy/n_e[0],Internal_Energy);

if ((OUTPUTFILE=fopen("CRM_Maxwellian_output.txt", "w"))==NULL)
        {
        printf("Cannot open file! Error!\n");
        exit(2);
       	}
//Main RK-4 loop
for (idx_t=1;idx_t<=t_iterations;idx_t++)
  {	T_e[idx_t]=T_e[idx_t-1];
	n_e[idx_t]=n_e[idx_t-1];
	solve_RK4_maxwellian(states_number, ionizations_number, excitations_number, delta_time, charge_vector, N, N_temp1, N_temp2, N_temp3, N_temp4, IntE_temp, n_e+idx_t, T_e+idx_t, &Internal_Energy,h_datapoints,h_w,h_x, h_j, h_k, h_l, T_r, excitations_indices, ionizations_indices,E_i,E_j,A_vector, B_vector1, C_vector, D_vector, R_1, R_2);
	if (idx_t%output_frequency==0){
		fprintf(OUTPUTFILE,"%E ",idx_t*delta_time); for(idx=0;idx<states_number;idx++){fprintf(OUTPUTFILE,"%E ",N[idx]);} fprintf(OUTPUTFILE,"\n");
			}
  }

//Output
for (idx_t=0;idx_t<=t_iterations;idx_t+=output_frequency)
	{
	fprintf(OUTPUTFILE,"%E %E %E\n",idx_t*delta_time,T_e[idx_t],n_e[idx_t]);
	}

exit(0);
}
