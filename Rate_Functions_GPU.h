#ifndef RATE_FUNCTIONS_GPU
#define RATE_FUNCTIONS_GPU

#define ACC_P 5
#define ACC_L 6503.0
#define ACC_IB1 6.5
#define ACC_IB2 15.0

__device__ double d_j_int(double E0,double E_j,double *B_vector);

__device__ double d_k_int(double E0,double E1,double E_i,double *C_vector);

__device__ double d_l_int(double EGamma,double T_r,double *D_vector);

__global__ void d_j_calc(double *d_params, double *E_j, double *B_vector, double *d_j_up, double *w, double *x);

__global__ void d_k_calc(double *d_params, double *E_i, double *C_vector,double *d_k_up, double *d_k_up_part, double *w, double *x);

__global__ void d_l_calc(double *d_params, double *E_i, double *D_vector, double *d_l, double *d_le, double *w, double *x);

void d_setup(double **d_params, double **d_B_vector, double **d_C_vector, double **d_D_vector, double **d_E_j, double **d_E_i, double **d_j, double **d_k, double **d_k_part, double **d_l, double **d_x, double **d_w, double *B_vector, double *C_vector, double *D_vector, double *E_j, double *E_i, double T_r, double *h_x, double *h_w, int ionizations_number, int excitations_number, int h_datapoints, cudaStream_t *streams);

void d_cleanup(double *d_params, double *d_B_vector, double *d_C_vector, double *d_E_j, double *d_E_i, double *d_j, double *d_k, double *d_k_part, double *d_l, double *d_x, double *d_w);

void d_calculate_rates(double *d_params,double *d_B_vector, double *d_C_vector, double *d_D_vector, double *d_E_j, double *d_E_i, double *d_j, double *d_k, double *d_k_part, double *d_l, double *d_x, double *d_w,double *h_params, double *h_j,double *h_k,double *h_l,double *h_w, double *h_x,double *ib_E, double n_e, double T_r, double *charge_vector, double *N, int states_number, int ionizations_number,int excitations_number,int h_datapoints,cudaStream_t *streams);

void hd_solve_RK4(int states_number, int ionizations_number, int excitations_number, double delta, double *charge_vector, double *N, double *N_temp1, double *N_temp2, double *N_temp3, double *N_temp4, double *IntE_temp, double *n_e, double *T_e, double *T_F, double *Internal_Energy,int h_datapoints, double *h_w, double *h_x, double *h_j, double *h_k, double *h_l, double T_r, int *excitations_indices, int *ionizations_indices, double *E_i, double *E_j,double *A_vector, double *B_vector, double *C_vector, double *D_vector, double *R_1, double *R_2, double *h_params, double *d_params, double *d_B_vector, double *d_C_vector, double *d_D_vector, double *d_E_j, double *d_E_i, double *d_j, double *d_k, double *d_k_part, double *d_l, double *d_x, double *d_w,cudaStream_t *streams);

#endif

#ifndef ACC_P
#define ACC_P 5
#endif
