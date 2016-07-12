#ifndef RATE_FUNCTIONS_CPU
#define RATE_FUNCTIONS_CPU

void gauss_integration_setup32(int *datapoints, double **weights, double **x);
void gauss_integration_setup32_f(int *datapoints, float **weights, float **x);

void gauss_integration_setup_fromfile(int *datapoints, double **weights, double **x);
void gauss_integration_setup_fromfile_f(int *datapoints, float **weights, float **x);

double vector_sum(int length, double *vector);
float vector_sum_f(int length, float *vector);

double Get_n_e(int states_number, double *N, double *charge_vector);
float Get_n_e_f(int states_number, float *N, float *charge_vector);

double Fermi_Energy(double n);
float Fermi_Energy_f(float n);

double Get_Chemical_Potential(double T, double T_F);
float Get_Chemical_Potential_f(float T, float T_F);

double Get_C_V(double Eta);
float Get_C_V_f(float Eta);

double Invert_C_V(double u);
float Invert_C_V_f(float u);

double h_j_int(double E0,double E_j,double *B_vector);
double h_k_int(double E0,double E1,double E_i,double *C_vector);
double h_l_int(double EGamma,double E_i, double T_r,double *D_vector);
double h_ib_int(double EGamma,double T_e,double mu,double T_r);

float h_j_int_f(float E0,float E_j,float *B_vector);
float h_k_int_f(float E0,float E1,float E_i,float *C_vector);
float h_l_int_f(float EGamma,float E_i, float T_r,float *D_vector);
float h_ib_int_f(float EGamma,float T_e,float mu,float T_r);

void h_j_gauss_integration(double T_e,double mu,double E_j,double *B_vector, int datapoints, double *h_j_up, double *weights, double *x);
void h_k_gauss_integration(double T_e,double mu,double E_i,double *C_vector, int datapoints, double *k_up, double *weights, double *x);
void h_l_gauss_integration(double T_e,double mu,double E_i,double T_r,double *D_vector, int datapoints, double *h_l, double *h_le, double *weights, double *x);
void h_j_gauss_integration_full(int levels_number,double T_e,double mu,double *E_j,double *B_vector, int datapoints, double *h_j, double *weights, double *x);
void h_k_gauss_integration_full(int ionizations_number,double T_e,double mu,double *E_i,double *C_vector, int datapoints, double *h_k, double *weights, double *x);
void h_l_gauss_integration_full(int ionizations_number,double T_e,double mu,double T_r,double *E_i,double *D_vector, int datapoints, double *h_l, double *weights, double *x);
double h_ib_gauss_integration(double T_e, double n_e,double mu,double T_r, int states_number, double *charge_vector, double *N, int datapoints, double *weights, double *x);

void h_j_gauss_integration_f(float T_e,float mu,float E_j,float *B_vector, int datapoints, float *h_j_up, float *weights, float *x);
void h_k_gauss_integration_f(float T_e,float mu,float E_i,float *C_vector, int datapoints, float *k_up, float *weights, float *x);
void h_l_gauss_integration_f(float T_e,float mu,float E_i,float T_r,float *D_vector, int datapoints, float *h_l, float *h_le, float *weights, float *x);
void h_j_gauss_integration_full_f(int excitations_number,float T_e,float mu,float *E_j,float *B_vector, int datapoints, float *h_j, float *weights, float *x);
void h_k_gauss_integration_full_f(int ionizations_number,float T_e,float mu,float *E_i,float *C_vector, int datapoints, float *h_k, float *weights, float *x);
void h_l_gauss_integration_full_f(int ionizations_number,float T_e,float mu,float T_r,float *E_i,float *D_vector, int datapoints, float *h_l, float *weights, float *x);
float h_ib_gauss_integration_f(float T_e, float n_e,float mu,float T_r, int states_number, float *charge_vector, float *N, int datapoints, float *weights, float *x);

void h_allocate_arrays(int states_number, int ionizations_number, int excitations_number, int h_datapoints, double **h_params, double **charge_vector, double **E_i,double **E_j,double **A_vector,double **B_vector, double **C_vector, double **D_vector, double **h_j, double **h_k, double **h_l, double **N, double **N_temp1, double **N_temp2, double **N_temp3, double **N_temp4, double **IntE_temp, double **R_1, double **R_2, int **excitations_indices, int **ionizations_indices);
void h_allocate_arrays_f(int states_number, int ionizations_number, int excitations_number, int h_datapoints, float **h_params, float **charge_vector, float **E_i,float **E_j,float **A_vector,float **B_vector, float **C_vector, float **D_vector, float **h_j, float **h_k, float **h_l, float **N, float **N_temp1, float **N_temp2, float **N_temp3, float **N_temp4, float **IntE_temp, float **R_1, float **R_2, int **excitations_indices, int **ionizations_indices);

void h_setup_atomic_model(int ionizations_number, int excitations_number,int *excitations_indices,int *ionizations_indices, double *charge_vector, double *E_i, double *E_j, double *A_vector, double *B_vector, double *C_vector, double *D_vector);
void h_setup_atomic_model_f(int ionizations_number, int excitations_number,int *excitations_indices,int *ionizations_indices, float *charge_vector, float *E_i, float *E_j, float *A_vector, float *B_vector, float *C_vector, float *D_vector);

void h_create_rate_matrices(int states_number, int ionizations_number, int excitations_number, double T_e, double n_e, double mu, double *R_1, double *R_2, double *j, double *k, double *l, int *excitations_indices, int *ionizations_indices, double *E_i, double *E_j);
void h_create_rate_matrices_f(int states_number, int ionizations_number, int excitations_number, float T_e, float n_e, float mu, float *R_1, float *R_2, float *j, float *k, float *l, int *excitations_indices, int *ionizations_indices, float *E_i, float *E_j);

void h_solve_RK4(int states_number, int ionizations_number, int excitations_number, double delta, double *charge_vector, double *N, double *N_temp1, double *N_temp2, double *N_temp3, double *N_temp4, double *IntE_temp, double *n_e, double *T_e, double *T_F, double *Internal_Energy,int h_datapoints, double *h_w, double *h_x, double *h_j, double *h_k, double *h_l, double T_r, int *excitations_indices, int *ionizations_indices, double *E_i, double *E_j,double *A_vector, double *B_vector, double *C_vector, double *D_vector, double *R_1, double *R_2);
void h_solve_RK4_f(int states_number, int ionizations_number, int excitations_number, float delta, float *charge_vector, float *N, float *N_temp1, float *N_temp2, float *N_temp3, float *N_temp4, float *IntE_temp, float *n_e, float *T_e, float *T_F, float *Internal_Energy,int h_datapoints, float *h_w, float *h_x, float *h_j, float *h_k, float *h_l, float T_r, int *excitations_indices, int *ionizations_indices, float *E_i, float *E_j,float *A_vector, float *B_vector, float *C_vector, float *D_vector, float *R_1, float *R_2);

void h_cleanup(double *h_params, double *charge_vector, double *E_i,double *E_j,double *A_vector,double *B_vector, double *C_vector, double *D_vector, double *h_j, double *h_k, double *h_l, double *h_w, double *h_x, double *N, double *N_temp1, double *N_temp2, double *N_temp3, double *N_temp4, double *IntE_temp, double *R_1, double *R_2, int *excitations_indices, int *ionizations_indices);
void h_cleanup_f(float *h_params, float *charge_vector, float *E_i,float *E_j,float *A_vector,float *B_vector, float *C_vector, float *D_vector, float *h_j, float *h_k, float *h_l, float *h_w, float *h_x, float *N, float *N_temp1, float *N_temp2, float *N_temp3, float *N_temp4, float *IntE_temp, float *R_1, float *R_2, int *excitations_indices, int *ionizations_indices);

double exp_int(double x);

double j_up1_maxwellian(double T_e,double E_j,double *B_vector);

double j_down_maxwellian(double j_up,double T_e,double E_j);

double k_up_maxwellian(double T_e,double E_i,double *C_vector);

double k_down_maxwellian(double k_up,double T_e,double E_i);

void l_up_maxwellian(double T_e,double E_i,double T_r,double *D_vector, int datapoints, double *h_l, double *h_le, double *weights, double *x);

void j_maxwellian_full(int excitations_number,double T_e,double *E_j,double *B_vector, double *h_j);

void k_maxwellian_full(int ionizations_number,double T_e,double *E_i,double *C_vector, double *h_k);

void l_maxwellian_full(int ionizations_number,double T_e,double T_r,double *E_i,double *D_vector, int datapoints, double *h_l, double *weights, double *x);

double ib_maxwellian(double T_e, double n_e,double T_r, int states_number, double *charge_vector, double *N, int datapoints, double *weights, double *x);

void create_rate_matrices_maxwellian(int states_number, int ionizations_number, int excitations_number, double T_e, double n_e, double *R_1, double *R_2, double *j, double *k, double *l, int *excitations_indices, int *ionizations_indices, double *E_i, double *E_j);

void solve_RK4_maxwellian(int states_number, int ionizations_number, int excitations_number, double delta, double *charge_vector, double *N, double *N_temp1, double *N_temp2, double *N_temp3, double *N_temp4, double *IntE_temp, double *n_e, double *T_e, double *Internal_Energy,int h_datapoints, double *h_w, double *h_x, double *h_j, double *h_k, double *h_l, double T_r, int *excitations_indices, int *ionizations_indices, double *E_i, double *E_j,double *A_vector, double *B_vector, double *C_vector, double *D_vector, double *R_1, double *R_2);

#endif
