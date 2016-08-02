#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <cblas.h>

extern "C" {  
#include "Rate_Functions_CPU.h"
}
#include "Rate_Functions_GPU.h"

#define ACC_J 19.013
#define ACC_K 25.253
#define ACC_L 6503.0

__constant__ int d_excitations_number,d_ionizations_number, d_datapoints, d_block_mult;
__constant__ double d_T_r;

void gauss_integration_setup32(double *weights, double *x)
{
	x[0]=1.3680690752591596E-03;
	x[1]=7.1942442273659202E-03;
	x[2]=1.7618872206246805E-02;
	x[3]=3.2546962031130167E-02;
	x[4]=5.1839422116973843E-02;
	x[5]=7.5316193133715015E-02;
	x[6]=1.0275810201602886E-01;
	x[7]=1.3390894062985509E-01;
	x[8]=1.6847786653489233E-01;
	x[9]=2.0614212137961868E-01;
	x[10]=2.4655004553388526E-01;
	x[11]=2.8932436193468253E-01;
	x[12]=3.3406569885893617E-01;
	x[13]=3.8035631887393162E-01;
	x[14]=4.2776401920860185E-01;
	x[15]=4.7584616715613093E-01;
	x[16]=5.2415383284386907E-01;
	x[17]=5.7223598079139815E-01;
	x[18]=6.1964368112606838E-01;
	x[19]=6.6593430114106378E-01;
	x[20]=7.1067563806531764E-01;
	x[21]=7.5344995446611462E-01;
	x[22]=7.9385787862038115E-01;
	x[23]=8.3152213346510750E-01;
	x[24]=8.6609105937014474E-01;
	x[25]=8.9724189798397114E-01;
	x[26]=9.2468380686628515E-01;
	x[27]=9.4816057788302599E-01;
	x[28]=9.6745303796886994E-01;
	x[29]=9.8238112779375319E-01;
	x[30]=9.9280575577263397E-01;
	x[31]=9.9863193092474067E-01;

	weights[0]=3.5093050047349198E-03;
	weights[1]=8.1371973654528751E-03;
	weights[2]=1.2696032654631021E-02;
	weights[3]=1.7136931456510726E-02;
	weights[4]=2.1417949011113720E-02;
	weights[5]=2.5499029631187890E-02;
	weights[6]=2.9342046739268091E-02;
	weights[7]=3.2911111388180682E-02;
	weights[8]=3.6172897054423871E-02;
	weights[9]=3.9096947893535162E-02;
	weights[10]=4.1655962113473763E-02;
	weights[11]=4.3826046502202044E-02;
	weights[12]=4.5586939347882056E-02;
	weights[13]=4.6922199540401971E-02;
	weights[14]=4.7819360039637472E-02;
	weights[15]=4.8270044257364274E-02;
	weights[16]=4.8270044257363830E-02;
	weights[17]=4.7819360039637784E-02;
	weights[18]=4.6922199540401846E-02;
	weights[19]=4.5586939347881918E-02;
	weights[20]=4.3826046502201850E-02;
	weights[21]=4.1655962113473798E-02;
	weights[22]=3.9096947893534850E-02;
	weights[23]=3.6172897054424745E-02;
	weights[24]=3.2911111388180932E-02;
	weights[25]=2.9342046739267064E-02;
	weights[26]=2.5499029631188164E-02;
	weights[27]=2.1417949011113362E-02;
	weights[28]=1.7136931456510799E-02;
	weights[29]=1.2696032654631212E-02;
	weights[30]=8.1371973654529653E-03;
	weights[31]=3.5093050047351631E-03;
}

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

//Carry out an integral for the collisional excitation coefficient
__global__ void d_j_calc(double *d_params, double *E_j, double *B_vector, double *d_j_up, double *w, double *x)
{	
	//Calculate the integrand on each thread
	int node_num, integral_num;
	integral_num=threadIdx.x/d_datapoints;
	node_num=threadIdx.x % d_datapoints;
	integral_num+=blockIdx.x*d_block_mult;
	extern  __shared__ double temp[];	//Integrand is stored in shared memory
	double *d_j_up_temp=&temp[0];
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

//Carry out a double integral for the collisional ionization coefficient
//Involves now two reductions and the intermediate results stored in main GPU memory
__global__ void d_k_calc(double *d_params, double *E_i, double *C_vector,double *d_k_up, double *w, double *x)
{	
	//Calculate the integrand on each thread
	int node_num, integral_num;
	integral_num=threadIdx.x/d_datapoints;
	node_num=threadIdx.x % d_datapoints;
	integral_num+=blockIdx.x*d_block_mult;
	extern  __shared__ double temp[];
	double *d_k_up_temp=&temp[0];
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

//Carry out an integral for the photoionization coefficient and energy change due to photoionization
//Single integral, similar to d_j_calc()
__global__ void d_l_calc(double *d_params, double *E_i, double *D_vector, double *d_l, double *d_le, double *w, double *x)
{	
	int node_num, integral_num;
	integral_num=threadIdx.x/d_datapoints;
	node_num=threadIdx.x % d_datapoints;
	integral_num+=blockIdx.x*d_block_mult;
	extern  __shared__ double temp[];
	double *d_l_temp=&temp[0];
	double *d_le_temp=&temp[blockDim.x];
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

void d_calculate_rates(double *d_params,double *d_B_vector, double *d_C_vector, double *d_D_vector, double *d_E_j, double *d_E_i, double *d_j, double *d_k, double *d_l, double *d_x, double *d_w,double *h_params, double *h_j,double *h_k,double *h_l,double *h_w, double *h_x,  double T_r, int ionizations_number,int excitations_number,int h_datapoints,cudaStream_t *streams, int h_block_mult)
{	
	dim3 block_dim(h_datapoints*h_block_mult,h_datapoints);
	cudaMemcpy(d_params,h_params,sizeof(double)*2,cudaMemcpyHostToDevice);
	d_j_calc<<<excitations_number/h_block_mult,h_datapoints*h_block_mult,h_datapoints*h_block_mult*sizeof(double),streams[0]>>>(d_params,d_E_j,d_B_vector,d_j,d_w,d_x);
	d_l_calc<<<ionizations_number/h_block_mult,h_datapoints*h_block_mult,2*h_datapoints*h_block_mult*sizeof(double),streams[0]>>>(d_params,d_E_i,d_D_vector,d_l,d_l+ionizations_number,d_w,d_x);
	d_k_calc<<<ionizations_number/h_block_mult,h_datapoints*h_block_mult,h_datapoints*h_block_mult*sizeof(double),streams[1]>>>(d_params,d_E_i,d_C_vector,d_k,d_w,d_x);
	cudaMemcpyAsync(h_j,d_j,sizeof(double)*excitations_number,cudaMemcpyDeviceToHost,streams[0]);
	cudaMemcpyAsync(h_l,d_l,2*sizeof(double)*ionizations_number,cudaMemcpyDeviceToHost,streams[0]);
	cudaMemcpyAsync(h_k,d_k,sizeof(double)*ionizations_number,cudaMemcpyDeviceToHost,streams[1]);
	cudaDeviceSynchronize();
}

//CPU memory allocation
void h_allocate_arrays(int ionizations_number, int excitations_number, int h_datapoints, double **h_params, double **E_i,double **E_j,double **B_vector, double **C_vector, double **D_vector, double **h_j, double **h_k, double **h_l, double **h_w, double **h_x)
{
	*h_params=(double*)malloc(2*sizeof(double));
	*E_i=(double*)malloc(ionizations_number*sizeof(double));
	*E_j=(double*)malloc(excitations_number*sizeof(double));
	*B_vector=(double*)malloc(excitations_number*4*sizeof(double));
	*C_vector=(double*)malloc(ionizations_number*5*sizeof(double));
	*D_vector=(double*)malloc(ionizations_number*2*sizeof(double));
	*h_j=(double*)malloc(excitations_number*sizeof(double));
	*h_k=(double*)malloc(ionizations_number*sizeof(double));
	*h_l=(double*)malloc(2*ionizations_number*sizeof(double));
	*h_x=(double*)malloc(h_datapoints*sizeof(double));		
	*h_w=(double*)malloc(h_datapoints*sizeof(double));
}

double h_j_int(double E0,double E_j,double *B_vector)
{
	double Eq=E_j/E0;
	double integrand=-log(Eq)*B_vector[0]+B_vector[1]+Eq*(B_vector[2]+Eq*B_vector[3]);
	return integrand;
}

//Evaluate the differential cross section for collisional ionization
//A Mott-type cross section, compatable with the BELI formula, is used
double h_k_int(double E0,double E1,double E_i,double *C_vector)
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

//Evaluate photoionization cross section
double h_l_int(double EGamma,double E_i, double T_r,double *D_vector)
{	double exp_EG=exp(EGamma/T_r)-1.0;
	double integrand=(D_vector[0]+D_vector[1]/EGamma)/exp_EG;
	return integrand;
}

//Full collisional excitation calculation
void h_j_gauss_integration(double T_e,double mu,double E_j,double *B_vector, int datapoints, double *h_j_up, double *weights, double *x)
{	double integrand=0.0, E0, fermi, fermi_m, integ_temp;
	double region_difference=(E_j+2.0*fabs(mu)+ACC_J*T_e);	
	int idx;


   for(idx=0;idx<datapoints;idx++)
	{
		E0=x[idx]*region_difference+E_j;
		integ_temp=h_j_int(E0,E_j,B_vector);
		fermi=1.0/(1.0+exp((E0-mu)/T_e));
		fermi_m=1.0/(1.0+exp((E0-E_j-mu)/T_e));
		integrand+=weights[idx]*integ_temp*fermi*(1.0-fermi_m);
	}

  *h_j_up=integrand*region_difference;

}

void h_k_gauss_integration(double T_e,double mu,double E_i,double *C_vector, int datapoints, double *k_up, double *weights, double *x)
{	double integrand0=0.0, integrand1, E0, E1, E0prime, fermiE0, fermiE1, fermiE0prime, integ_temp;
	double region_difference=(fabs(mu)+50.0+ACC_K*T_e);
	int idx0,idx1;

   for(idx0=0;idx0<datapoints;idx0++)
	{
	E0prime=x[idx0]*region_difference;
	E0=E0prime+E_i;
	integrand1=0.0;
	for(idx1=0;idx1<datapoints;idx1++)
	  {
		E1=x[idx1]*E0prime;
		integ_temp=h_k_int(E0, E1, E_i,C_vector)*weights[idx1];
		fermiE0=1.0/(1.0+exp((E0-mu)/T_e));
		fermiE1=1.0/(1.0+exp((E1-mu)/T_e));
		fermiE0prime=1.0/(1.0+exp((E0prime-E1-mu)/T_e));
		integrand1+=integ_temp*fermiE0*(1.0-fermiE1)*(1.0-fermiE0prime);
	  }
	integrand0+=weights[idx0]*E0prime*integrand1;
	}

  *k_up=integrand0*region_difference;
}

void h_l_gauss_integration(double T_e,double mu,double E_i,double T_r,double *D_vector, int datapoints, double *h_l, double *h_le, double *weights, double *x)
{	double integrand0=0.0, integrand1=0.0, EGamma, EGammaPrime, fermi_m, integ_temp;
	double region_difference=ACC_L;
	if (mu>0.0){region_difference+=mu;}
	int idx;

   for(idx=0;idx<datapoints;idx++)
	{
		EGammaPrime=x[idx]*region_difference;
		EGamma=EGammaPrime+E_i;
		fermi_m=1.0-1.0/(1.0+exp((EGammaPrime-mu)/T_e));
		integ_temp=h_l_int(EGamma,E_i,T_r,D_vector)*weights[idx]*fermi_m;
		integrand0+=integ_temp;
		integrand1+=integ_temp*EGammaPrime;
	}
  *h_l=integrand0*region_difference;
  *h_le=integrand1*region_difference;
	
}

//The following functions carry out sequential integration for all relevant states
void h_j_gauss_integration_full(int excitations_number,double T_e,double mu,double *E_j,double *B_vector, int datapoints, double *h_j, double *weights, double *x)
{ int idx_j;

  for (idx_j=0;idx_j<excitations_number;idx_j++)
	{
	h_j_gauss_integration(T_e,mu,E_j[idx_j],B_vector+idx_j*4,datapoints,h_j+idx_j,weights,x);
	}
}

void h_k_gauss_integration_full(int ionizations_number,double T_e,double mu,double *E_i,double *C_vector, int datapoints, double *h_k, double *weights, double *x)
{ int idx_k;

for  (idx_k=0;idx_k<ionizations_number;idx_k++)
	{
	h_k_gauss_integration(T_e,mu,E_i[idx_k],C_vector+idx_k*5,datapoints,h_k+idx_k,weights,x);
	}
}

void h_l_gauss_integration_full(int ionizations_number,double T_e,double mu,double T_r,double *E_i,double *D_vector, int datapoints, double *h_l, double *weights, double *x)
{ int idx_l;
for  (idx_l=0;idx_l<ionizations_number;idx_l++)
	{
	h_l_gauss_integration(T_e,mu,E_i[idx_l],T_r,D_vector+idx_l*2,datapoints,h_l+idx_l,h_l+ionizations_number+idx_l,weights,x);
	}
}

int main(int argc, char *argv[])
{

int h_datapoints=32, ionizations_number=10000, excitations_number=10000, idx, h_block_mult=1;
if (argc>1){ionizations_number=atoi(argv[1]);}
if (argc>2){excitations_number=atoi(argv[2]);}
if (argc>3){h_block_mult=atoi(argv[3]);}
double *h_params, *E_i, *E_j, *B_vector, *C_vector, *D_vector, T_r;
double *h_j, *h_k, *h_l, *h_x, *h_w, h_j2,h_k2,h_l2,h_le2;
FILE *INPUTFILE1, *INPUTFILE2;
clock_t h_start_t, h_end_t;
double h_total_t;

h_allocate_arrays(ionizations_number,excitations_number,h_datapoints,&h_params,&E_i,&E_j,&B_vector,&C_vector, &D_vector,&h_j, &h_k,&h_l,&h_w,&h_x);
gauss_integration_setup32(h_w,h_x);
h_params[0]=10.0;	h_params[1]=3.0;
T_r=300.0;

if ((INPUTFILE1=fopen("Test_Ionization_Coeffs.txt", "r"))==NULL)
        {
        printf("Cannot open file! Error!\n");
        exit(2);
       	}

for(idx=0;idx<ionizations_number;idx++)
	{
	fscanf(INPUTFILE1,"%lf %lf %lf %lf %lf %lf %lf %lf", &E_i[idx], &C_vector[idx*5], &C_vector[idx*5+1], &C_vector[idx*5+2], &C_vector[idx*5+3], &C_vector[idx*5+4], &D_vector[idx*2], &D_vector[idx*2+1]);
	}

fclose(INPUTFILE1);

if ((INPUTFILE2=fopen("Test_Excitation_Coeffs.txt", "r"))==NULL)
        {
        printf("Cannot open file! Error!\n");
        exit(2);
       	}

for(idx=0;idx<excitations_number;idx++)
	{
	fscanf(INPUTFILE2,"%lf %lf %lf %lf %lf", &E_j[idx], &B_vector[idx*4], &B_vector[idx*4+1], &B_vector[idx*4+2], &B_vector[idx*4+3]);
	}

fclose(INPUTFILE2);

int device_count=0;
cudaGetDeviceCount(&device_count);
printf("Device count: %i\n",device_count);
cudaSetDevice(0);	//Run on device 0 by default - can be changed if multiple GPUs etc are present
cudaStream_t streams[2];
float gpu_time, gpu_time1, gpu_time2;
cudaEvent_t start, stop, start1, stop1, start2, stop2;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventCreate(&start1);
cudaEventCreate(&stop1);
cudaEventCreate(&start2);
cudaEventCreate(&stop2);

double *d_params, *d_B_vector, *d_C_vector, *d_D_vector, *d_E_j, *d_E_i, *d_j, *d_k, *d_l, *d_x, *d_w;
d_setup(&d_params,&d_B_vector,&d_C_vector,&d_D_vector,&d_E_j,&d_E_i, &d_j, &d_k, &d_l, &d_x, &d_w,B_vector, C_vector, D_vector, E_j, E_i, T_r, h_x,h_w, ionizations_number,excitations_number, h_datapoints,streams,h_block_mult);

h_start_t=clock();
cudaEventRecord(start);
cudaEventRecord(start1,streams[0]);
cudaEventRecord(start2,streams[1]);

d_calculate_rates(d_params,d_B_vector, d_C_vector,d_D_vector, d_E_j, d_E_i, d_j, d_k, d_l, d_x,  d_w,h_params,h_j,h_k,h_l,h_w,h_x,T_r, ionizations_number,excitations_number,h_datapoints,streams,h_block_mult);

cudaEventRecord(stop);
cudaEventRecord(stop1,streams[0]);
cudaEventRecord(stop2,streams[1]);
cudaEventSynchronize(stop);
cudaEventSynchronize(stop1);
cudaEventSynchronize(stop2);
cudaEventElapsedTime(&gpu_time, start, stop);
cudaEventElapsedTime(&gpu_time1, start1, stop1);
cudaEventElapsedTime(&gpu_time2, start2, stop2);
h_end_t=clock();
h_total_t=(double)(h_end_t-h_start_t)/CLOCKS_PER_SEC;
printf("Time: %E   CUDA times: %f %f %f\n", h_total_t, gpu_time, gpu_time1, gpu_time2);

for(idx=0;idx<10;idx++)
	{
	h_j_gauss_integration(h_params[0],h_params[1],E_j[idx],B_vector+4*idx,h_datapoints,&h_j2, h_w, h_x);
	h_k_gauss_integration(h_params[0],h_params[1],E_i[idx],C_vector+5*idx,h_datapoints,&h_k2, h_w, h_x);
	h_l_gauss_integration(h_params[0],h_params[1],E_i[idx],T_r,D_vector+2*idx,h_datapoints,&h_l2, &h_le2, h_w, h_x);
	printf("%E %E %E %E %E %E %E %E\n",h_j[idx],h_j2,h_k[idx],h_k2,h_l[idx],h_l2,h_l[idx+ionizations_number],h_le2);
	}

h_j_gauss_integration(h_params[0],h_params[1],E_j[excitations_number-1],B_vector+4*(excitations_number-1),h_datapoints,&h_j2, h_w, h_x);
h_k_gauss_integration(h_params[0],h_params[1],E_i[ionizations_number-1],C_vector+5*(ionizations_number-1),h_datapoints,&h_k2, h_w, h_x);
h_l_gauss_integration(h_params[0],h_params[1],E_i[ionizations_number-1],T_r,D_vector+2*(ionizations_number-1),h_datapoints,&h_l2, &h_le2, h_w, h_x);
printf("%E %E %E %E %E %E %E %E\n",h_j[excitations_number-1],h_j2,h_k[ionizations_number-1],h_k2,h_l[ionizations_number-1],h_l2,h_l[2*ionizations_number-1],h_le2);


d_cleanup(d_params, d_B_vector, d_C_vector, d_E_j, d_E_i, d_j, d_k, d_l, d_x, d_w);
exit(0);
}


