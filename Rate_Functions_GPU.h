#ifndef RATE_FUNCTIONS_GPU
#define RATE_FUNCTIONS_GPU

__device__ double d_j_int(double E0,double E_j,double *B_vector);
__device__ double d_k_int(double E0,double E1,double E_i,double *C_vector);
__device__ double d_l_int(double EGamma,double T_r,double *D_vector);

__device__ float d_j_int_f(float E0,float E_j,float *B_vector);
__device__ float d_k_int_f(float E0,float E1,float E_i,float *C_vector);
__device__ float d_l_int_f(float EGamma,float T_r,float *D_vector);

__global__ void d_j_calc(double *d_params, double *E_j, double *B_vector, double *d_j_up, double *w, double *x);
__global__ void d_j_calc_f(float *d_params, float *E_j, float *B_vector, float *d_j_up, float *w, float *x);

__global__ void d_k_calc(double *d_params, double *E_i, double *C_vector,double *d_k_up, double *w, double *x);
__global__ void d_k_calc_f(float *d_params, float *E_i, float *C_vector,float *d_k_up, float *w, float *x);

__global__ void d_l_calc(double *d_params, double *E_i, double *D_vector, double *d_l, double *d_le, double *w, double *x);
__global__ void d_l_calc_f(float *d_params, float *E_i, float *D_vector, float *d_l, float *d_le, float *w, float *x);

void d_setup(double **d_params, double **d_B_vector, double **d_C_vector, double **d_D_vector, double **d_E_j, double **d_E_i, double **d_j, double **d_k, double **d_l, double **d_x, double **d_w, double *B_vector, double *C_vector, double *D_vector, double *E_j, double *E_i, double T_r, double *h_x, double *h_w, int ionizations_number, int excitations_number, int h_datapoints, cudaStream_t *streams, int h_block_mult);
void d_setup_f(float **d_params, float **d_B_vector, float **d_C_vector, float **d_D_vector, float **d_E_j, float **d_E_i, float **d_j, float **d_k, float **d_l, float **d_x, float **d_w, float *B_vector, float *C_vector, float *D_vector, float *E_j, float *E_i, float T_r, float *h_x, float *h_w, int ionizations_number, int excitations_number, int h_datapoints, cudaStream_t *streams, int h_block_mult);

void d_cleanup(double *d_params, double *d_B_vector, double *d_C_vector, double *d_E_j, double *d_E_i, double *d_j, double *d_k, double *d_l, double *d_x, double *d_w);
void d_cleanup_f(float *d_params, float *d_B_vector, float *d_C_vector, float *d_E_j, float *d_E_i, float *d_j, float *d_k, float *d_l, float *d_x, float *d_w);

void d_calculate_rates(double *d_params,double *d_B_vector, double *d_C_vector, double *d_D_vector, double *d_E_j, double *d_E_i, double *d_j, double *d_k, double *d_l, double *d_x, double *d_w,double *h_params, double *h_j,double *h_k,double *h_l,double *h_w, double *h_x, double  *ib_E, double n_e, double T_r, double *charge_vector, double *N, int states_number, int ionizations_number,int excitations_number,int h_datapoints,cudaStream_t *streams, int h_block_mult);
void d_calculate_rates_f(float *d_params,float *d_B_vector, float *d_C_vector, float *d_D_vector, float *d_E_j, float *d_E_i, float *d_j, float *d_k, float *d_l, float *d_x, float *d_w,float *h_params, float *h_j,float *h_k,float *h_l,float *h_w, float *h_x, float  *ib_E, float n_e, float T_r, float *charge_vector, float *N, int states_number, int ionizations_number,int excitations_number,int h_datapoints,cudaStream_t *streams, int h_block_mult);

void hd_solve_RK4(int states_number, int ionizations_number, int excitations_number, double delta, double *charge_vector, double *N, double *N_temp1, double *N_temp2, double *N_temp3, double *N_temp4, double *IntE_temp, double *n_e, double *T_e, double *T_F, double *Internal_Energy,int h_datapoints, double *h_w, double *h_x, double *h_j, double *h_k, double *h_l, double T_r, int *excitations_indices, int *ionizations_indices, double *E_i, double *E_j,double *A_vector, double *B_vector, double *C_vector, double *D_vector, double *R_1, double *R_2, double *h_params, double *d_params, double *d_B_vector, double *d_C_vector, double *d_D_vector, double *d_E_j, double *d_E_i, double *d_j, double *d_k, double *d_k_part, double *d_l, double *d_x, double *d_w,cudaStream_t *streams);
void hd_solve_RK4_f(int states_number, int ionizations_number, int excitations_number, float delta, float *charge_vector, float *N, float *N_temp1, float *N_temp2, float *N_temp3, float *N_temp4, float *IntE_temp, float *n_e, float *T_e, float *T_F, float *Internal_Energy,int h_datapoints, float *h_w, float *h_x, float *h_j, float *h_k, float *h_l, float T_r, int *excitations_indices, int *ionizations_indices, float *E_i, float *E_j,float *A_vector, float *B_vector, float *C_vector, float *D_vector, float *R_1, float *R_2, float *h_params, float *d_params, float *d_B_vector, float *d_C_vector, float *d_D_vector, float *d_E_j, float *d_E_i, float *d_j, float *d_k, float *d_l, float *d_x, float *d_w,cudaStream_t *streams, int h_block_mult);

#endif
