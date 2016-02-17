#ifndef RATE_FUNCTIONS_CPU
#define RATE_FUNCTIONS_CPU

double vector_sum(int length, double *vector);

double Get_n_e(int states_number, double *N, double *charge_vector);

double Fermi_Energy(double n);

double Get_Chemical_Potential(double T, double T_F);

double Get_C_V(double Eta);

double Invert_C_V(double u);

double h_j_int(double E0,double E_j,double *B_vector);

double h_k_int(double E0,double E1,double E_i,double *C_vector);

double h_l_int(double EGamma,double E_i, double T_r,double *D_vector);

double h_ib_int(double EGamma,double T_e,double mu,double T_r);

void gauss_integration_setup(int datapoints, double *weights, double *x,double *coeffs, double *vectors);

void h_j_gauss_integration(double T_e,double mu,double E_j,double *B_vector, int datapoints, double *h_j_up, double *weights, double *x);

void h_k_gauss_integration(double T_e,double mu,double E_i,double *C_vector, int datapoints, double *k_up, double *weights, double *x);

void h_l_gauss_integration(double T_e,double mu,double E_i,double T_r,double *D_vector, int datapoints, double *h_l, double *h_le, double *weights, double *x);

void h_j_gauss_integration_full(int levels_number,double T_e,double mu,double *E_j,double *B_vector, int datapoints, double *h_j, double *weights, double *x);

void h_k_gauss_integration_full(int ionizations_number,double T_e,double mu,double *E_i,double *C_vector, int datapoints, double *h_k, double *weights, double *x);

void h_l_gauss_integration_full(int ionizations_number,double T_e,double mu,double T_r,double *E_i,double *D_vector, int datapoints, double *h_l, double *weights, double *x);

double h_ib_gauss_integration(double T_e, double n_e,double mu,double T_r, int states_number, double *charge_vector, double *N, int datapoints, double *weights, double *x);

void h_allocate_arrays(int states_number, int ionizations_number, int excitations_number, int h_datapoints, double **h_params, double **charge_vector, double **E_i,double **E_j,double **A_vector,double **B_vector1, double **B_vector2, double **C_vector, double **D_vector, double **h_j, double **h_k, double **h_l, double **h_w, double **h_x, double **h_c, double **h_v, double **N, double **N_temp1, double **N_temp2, double **N_temp3, double **N_temp4, double **IntE_temp, double **R_1, double **R_2, int **excitations_indices, int **ionizations_indices);

void h_setup_atomic_model(int ionizations_number, int excitations_number,int *excitations_indices,int *ionizations_indices, double *charge_vector, double *E_i, double *E_j, double *A_vector, double *B_vector1, double *C_vector, double *D_vector);

void h_create_rate_matrices(int states_number, int ionizations_number, int excitations_number, double T_e, double n_e, double mu, double *R_1, double *R_2, double *j, double *k, double *l, int *excitations_indices, int *ionizations_indices, double *E_i, double *E_j);

void h_solve_RK4(int states_number, int ionizations_number, int excitations_number, double delta, double *charge_vector, double *N, double *N_temp1, double *N_temp2, double *N_temp3, double *N_temp4, double *IntE_temp, double *n_e, double *T_e, double *T_F, double *Internal_Energy,int h_datapoints, double *h_w, double *h_x, double *h_j, double *h_k, double *h_l, double T_r, int *excitations_indices, int *ionizations_indices, double *E_i, double *E_j,double *A_vector, double *B_vector1, double *C_vector, double *D_vector, double *R_1, double *R_2);

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

void solve_RK4_maxwellian(int states_number, int ionizations_number, int excitations_number, double delta, double *charge_vector, double *N, double *N_temp1, double *N_temp2, double *N_temp3, double *N_temp4, double *IntE_temp, double *n_e, double *T_e, double *Internal_Energy,int h_datapoints, double *h_w, double *h_x, double *h_j, double *h_k, double *h_l, double T_r, int *excitations_indices, int *ionizations_indices, double *E_i, double *E_j,double *A_vector, double *B_vector1, double *C_vector, double *D_vector, double *R_1, double *R_2);

#endif

#ifndef ACC_P
#define ACC_P 5
#endif
