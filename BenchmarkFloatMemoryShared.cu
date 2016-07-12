#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

#define ACC_Jf 19.013f
#define ACC_Kf 25.253f
#define ACC_Lf 6503.0f

__constant__ int d_excitations_number,d_ionizations_number, d_datapoints, d_block_mult;
__constant__ float d_T_r_f;

void gauss_integration_setup32_f(float *weights, float *x)
{
	x[0]=1.3680690752591596E-03f;
	x[1]=7.1942442273659202E-03f;
	x[2]=1.7618872206246805E-02f;
	x[3]=3.2546962031130167E-02f;
	x[4]=5.1839422116973843E-02f;
	x[5]=7.5316193133715015E-02f;
	x[6]=1.0275810201602886E-01f;
	x[7]=1.3390894062985509E-01f;
	x[8]=1.6847786653489233E-01f;
	x[9]=2.0614212137961868E-01f;
	x[10]=2.4655004553388526E-01f;
	x[11]=2.8932436193468253E-01f;
	x[12]=3.3406569885893617E-01f;
	x[13]=3.8035631887393162E-01f;
	x[14]=4.2776401920860185E-01f;
	x[15]=4.7584616715613093E-01f;
	x[16]=5.2415383284386907E-01f;
	x[17]=5.7223598079139815E-01f;
	x[18]=6.1964368112606838E-01f;
	x[19]=6.6593430114106378E-01f;
	x[20]=7.1067563806531764E-01f;
	x[21]=7.5344995446611462E-01f;
	x[22]=7.9385787862038115E-01f;
	x[23]=8.3152213346510750E-01f;
	x[24]=8.6609105937014474E-01f;
	x[25]=8.9724189798397114E-01f;
	x[26]=9.2468380686628515E-01f;
	x[27]=9.4816057788302599E-01f;
	x[28]=9.6745303796886994E-01f;
	x[29]=9.8238112779375319E-01f;
	x[30]=9.9280575577263397E-01f;
	x[31]=9.9863193092474067E-01f;

	weights[0]=3.5093050047349198E-03f;
	weights[1]=8.1371973654528751E-03f;
	weights[2]=1.2696032654631021E-02f;
	weights[3]=1.7136931456510726E-02f;
	weights[4]=2.1417949011113720E-02f;
	weights[5]=2.5499029631187890E-02f;
	weights[6]=2.9342046739268091E-02f;
	weights[7]=3.2911111388180682E-02f;
	weights[8]=3.6172897054423871E-02f;
	weights[9]=3.9096947893535162E-02f;
	weights[10]=4.1655962113473763E-02f;
	weights[11]=4.3826046502202044E-02f;
	weights[12]=4.5586939347882056E-02f;
	weights[13]=4.6922199540401971E-02f;
	weights[14]=4.7819360039637472E-02f;
	weights[15]=4.8270044257364274E-02f;
	weights[16]=4.8270044257363830E-02f;
	weights[17]=4.7819360039637784E-02f;
	weights[18]=4.6922199540401846E-02f;
	weights[19]=4.5586939347881918E-02f;
	weights[20]=4.3826046502201850E-02f;
	weights[21]=4.1655962113473798E-02f;
	weights[22]=3.9096947893534850E-02f;
	weights[23]=3.6172897054424745E-02f;
	weights[24]=3.2911111388180932E-02f;
	weights[25]=2.9342046739267064E-02f;
	weights[26]=2.5499029631188164E-02f;
	weights[27]=2.1417949011113362E-02f;
	weights[28]=1.7136931456510799E-02f;
	weights[29]=1.2696032654631212E-02f;
	weights[30]=8.1371973654529653E-03f;
	weights[31]=3.5093050047351631E-03f;
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
__global__ void d_j_calc_f(float *d_params, float *E_j, float *B_vector, float *d_j_up, float *w, float *x)
{	
	//Calculate the integrand on each thread
	int node_num, integral_num;
	integral_num=threadIdx.x/d_datapoints;
	node_num=threadIdx.x % d_datapoints;
	integral_num+=blockIdx.x*d_block_mult;
	extern  __shared__ float temp[];	//Integrand is stored in shared memory
	float *d_j_up_temp=&temp[0];
	float lim=(E_j[integral_num]+2.0f*fabsf(d_params[1])+ACC_Jf*d_params[0]);
	float E0=x[node_num]*lim+E_j[integral_num];
	float integrand=d_j_int_f(E0,E_j[integral_num],B_vector+integral_num*4)*w[node_num];
	float fermi=1.0f/(1.0f+expf((E0-d_params[1])/d_params[0]));
	float fermi_m=1.0f/(1.0f+expf((E0-E_j[integral_num]-d_params[1])/d_params[0]));
	d_j_up_temp[threadIdx.x]=integrand*fermi*(1.0f-fermi_m);

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
__global__ void d_k_calc_fmem(float *d_params, float *E_i, float *C_vector,float *d_k_up, float *d_k_up_part, float *w, float *x)
{	
	//Calculate the integrand on each thread
	int node_num, integral_num;
	integral_num=threadIdx.x/d_datapoints;
	node_num=threadIdx.x % d_datapoints;
	integral_num+=blockIdx.x*d_block_mult;
	extern  __shared__ float temp[];
	float *d_k_up_temp=&temp[0];
	float lim=(fabsf(d_params[1])+50.0f+ACC_Kf*d_params[0]);
	float E0prime=x[node_num]*lim;
	float E0=E0prime+E_i[integral_num];
	float E1=x[blockIdx.x]*E0prime;
	float integrand=d_k_int_f(E0,E1,E_i[integral_num],C_vector+integral_num*5);
	float fermiE0=1.0f/(1.0f+exp((E0-d_params[1])/d_params[0]));
	float fermiE1=1.0f/(1.0f+exp((E1-d_params[1])/d_params[0]));
	float fermiE0prime=1.0f/(1.0f+exp((E0prime-E1-d_params[1])/d_params[0]));
	float int_w_w=integrand*w[node_num]*w[blockIdx.x]*E0prime;
	d_k_up_temp[threadIdx.x]=int_w_w*fermiE0*(1.0f-fermiE1)*(1.0f-fermiE0prime);

	//Series of reductions over the first integral
    for (int d = d_datapoints>>1; d > 0; d >>= 1)
    {
      __syncthreads(); 
      if (node_num<d)
	{
		d_k_up_temp[threadIdx.x]   += d_k_up_temp[threadIdx.x+d];
	}
    }
	__syncthreads();
	
	//The results of the first integrals are stored in main GPU memory
    if (node_num==0) 
    {	
	d_k_up_part[blockIdx.x+gridDim.x*integral_num]=d_k_up_temp[threadIdx.x]*lim;
    }
	int tid=node_num+gridDim.x*integral_num;

	__syncthreads();

	//Reduction over the second integral to obtain a value for the full double integral
    if(blockIdx.x==0)
    {
	d_k_up_temp[threadIdx.x]=d_k_up_part[tid];
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
		d_k_up[integral_num]=d_k_up_temp[threadIdx.x];
	    }
    }
}

//Carry out an integral for the photoionization coefficient and energy change due to photoionization
//Single integral, similar to d_j_calc()
__global__ void d_l_calc_f(float *d_params, float *E_i, float *D_vector, float *d_l, float *d_le, float *w, float *x)
{	
	int node_num, integral_num;
	integral_num=threadIdx.x/d_datapoints;
	node_num=threadIdx.x % d_datapoints;
	integral_num+=blockIdx.x*d_block_mult;
	extern  __shared__ float temp[];
	float *d_l_temp=&temp[0];
	float *d_le_temp=&temp[blockDim.x];
	float lim=ACC_Lf;
	if(d_params[1]>0.0f){lim+=d_params[1];}
	float EGammaPrime=x[node_num]*lim;
	float EGamma=EGammaPrime+E_i[integral_num];
	float fermi_m=1.0f-1.0f/(1.0f+expf((EGammaPrime-d_params[1])/d_params[0]));
	float integrand=d_l_int_f(EGamma,d_T_r_f,D_vector+integral_num*2)*w[node_num]*fermi_m;
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

void d_calculate_rates_fmem(float *d_params,float *d_B_vector, float *d_C_vector, float *d_D_vector, float *d_E_j, float *d_E_i, float *d_j, float *d_k, float *d_k_part, float *d_l, float *d_x, float *d_w,float *h_params, float *h_j,float *h_k,float *h_l,float *h_w, float *h_x,  float T_r, int ionizations_number,int excitations_number,int h_datapoints,cudaStream_t *streams, int h_block_mult)
{	
	dim3 grid_dim(h_datapoints,ionizations_number);
	cudaMemcpy(d_params,h_params,sizeof(float)*2,cudaMemcpyHostToDevice);
	d_j_calc_f<<<excitations_number/h_block_mult,h_datapoints*h_block_mult,h_datapoints*h_block_mult*sizeof(float),streams[0]>>>(d_params,d_E_j,d_B_vector,d_j,d_w,d_x);
	d_l_calc_f<<<ionizations_number/h_block_mult,h_datapoints*h_block_mult,2*h_datapoints*h_block_mult*sizeof(float),streams[0]>>>(d_params,d_E_i,d_D_vector,d_l,d_l+ionizations_number,d_w,d_x);
	d_k_calc_fmem<<<ionizations_number/h_block_mult,h_datapoints*h_block_mult,h_datapoints*h_block_mult*sizeof(float),streams[1]>>>(d_params,d_E_i,d_C_vector,d_k,d_k_part,d_w,d_x);
	cudaMemcpyAsync(h_j,d_j,sizeof(float)*excitations_number,cudaMemcpyDeviceToHost,streams[0]);
	cudaMemcpyAsync(h_l,d_l,2*sizeof(float)*ionizations_number,cudaMemcpyDeviceToHost,streams[0]);
	cudaMemcpyAsync(h_k,d_k,sizeof(float)*ionizations_number,cudaMemcpyDeviceToHost,streams[1]);
	cudaDeviceSynchronize();
}

//CPU memory allocation
void h_allocate_arrays_f(int ionizations_number, int excitations_number, int h_datapoints, float **h_params, float **E_i,float **E_j,float **B_vector, float **C_vector, float **D_vector, float **h_j, float **h_k, float **h_l, float **h_w, float **h_x)
{
	*h_params=(float*)malloc(2*sizeof(float));
	*E_i=(float*)malloc(ionizations_number*sizeof(float));
	*E_j=(float*)malloc(excitations_number*sizeof(float));
	*B_vector=(float*)malloc(excitations_number*4*sizeof(float));
	*C_vector=(float*)malloc(ionizations_number*5*sizeof(float));
	*D_vector=(float*)malloc(ionizations_number*2*sizeof(float));
	*h_j=(float*)malloc(excitations_number*sizeof(float));
	*h_k=(float*)malloc(ionizations_number*sizeof(float));
	*h_l=(float*)malloc(2*ionizations_number*sizeof(float));
	*h_x=(float*)malloc(h_datapoints*sizeof(float));		
	*h_w=(float*)malloc(h_datapoints*sizeof(float));
}

float h_j_int_f(float E0,float E_j,float *B_vector)
{
	float Eq=E_j/E0;
	float integrand=-logf(Eq)*B_vector[0]+B_vector[1]+Eq*(B_vector[2]+Eq*B_vector[3]);
	return integrand;
}

//Evaluate the differential cross section for collisional ionization
//A Mott-type cross section, compatable with the BELI formula, is used
float h_k_int_f(float E0,float E1,float E_i,float *C_vector)
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

//Evaluate photoionization cross section
float h_l_int_f(float EGamma,float E_i, float T_r,float *D_vector)
{	float exp_EG=expf(EGamma/T_r)-1.0f;
	float integrand=(D_vector[0]+D_vector[1]/EGamma)/exp_EG;
	return integrand;
}

//Full collisional excitation calculation
void h_j_gauss_integration_f(float T_e,float mu,float E_j,float *B_vector, int datapoints, float *h_j_up, float *weights, float *x)
{	float integrand=0.0f, E0, fermi, fermi_m, integ_temp;
	float region_difference=(E_j+2.0f*fabsf(mu)+ACC_Jf*T_e);	
	int idx;


   for(idx=0;idx<datapoints;idx++)
	{
		E0=x[idx]*region_difference+E_j;
		integ_temp=h_j_int_f(E0,E_j,B_vector);
		fermi=1.0f/(1.0f+expf((E0-mu)/T_e));
		fermi_m=1.0f/(1.0f+expf((E0-E_j-mu)/T_e));
		integrand+=weights[idx]*integ_temp*fermi*(1.0f-fermi_m);
	}

  *h_j_up=integrand*region_difference;

}

void h_k_gauss_integration_f(float T_e,float mu,float E_i,float *C_vector, int datapoints, float *k_up, float *weights, float *x)
{	float integrand0=0.0f, integrand1, E0, E1, E0prime, fermiE0, fermiE1, fermiE0prime, integ_temp;
	float region_difference=(fabsf(mu)+50.0f+ACC_Kf*T_e);
	int idx0,idx1;

   for(idx0=0;idx0<datapoints;idx0++)
	{
	E0prime=x[idx0]*region_difference;
	E0=E0prime+E_i;
	integrand1=0.0f;
	for(idx1=0;idx1<datapoints;idx1++)
	  {
		E1=x[idx1]*E0prime;
		integ_temp=h_k_int_f(E0, E1, E_i,C_vector)*weights[idx1];
		fermiE0=1.0f/(1.0f+expf((E0-mu)/T_e));
		fermiE1=1.0f/(1.0f+expf((E1-mu)/T_e));
		fermiE0prime=1.0f/(1.0f+expf((E0prime-E1-mu)/T_e));
		integrand1+=integ_temp*fermiE0*(1.0f-fermiE1)*(1.0f-fermiE0prime);
	  }
	integrand0+=weights[idx0]*E0prime*integrand1;
	}

  *k_up=integrand0*region_difference;
}

void h_l_gauss_integration_f(float T_e,float mu,float E_i,float T_r,float *D_vector, int datapoints, float *h_l, float *h_le, float *weights, float *x)
{	float integrand0=0.0f, integrand1=0.0f, EGamma, EGammaPrime, fermi_m, integ_temp;
	float region_difference=ACC_Lf;
	if (mu>0.0f){region_difference+=mu;}
	int idx;

   for(idx=0;idx<datapoints;idx++)
	{
		EGammaPrime=x[idx]*region_difference;
		EGamma=EGammaPrime+E_i;
		fermi_m=1.0f-1.0f/(1.0f+expf((EGammaPrime-mu)/T_e));
		integ_temp=h_l_int_f(EGamma,E_i,T_r,D_vector)*weights[idx]*fermi_m;
		integrand0+=integ_temp;
		integrand1+=integ_temp*EGammaPrime;
	}
  *h_l=integrand0*region_difference;
  *h_le=integrand1*region_difference;
	
}

//The following functions carry out sequential integration for all relevant states
void h_j_gauss_integration_full_f(int excitations_number,float T_e,float mu,float *E_j,float *B_vector, int datapoints, float *h_j, float *weights, float *x)
{ int idx_j;

  for (idx_j=0;idx_j<excitations_number;idx_j++)
	{
	h_j_gauss_integration_f(T_e,mu,E_j[idx_j],B_vector+idx_j*4,datapoints,h_j+idx_j,weights,x);
	}
}

void h_k_gauss_integration_full_f(int ionizations_number,float T_e,float mu,float *E_i,float *C_vector, int datapoints, float *h_k, float *weights, float *x)
{ int idx_k;

for  (idx_k=0;idx_k<ionizations_number;idx_k++)
	{
	h_k_gauss_integration_f(T_e,mu,E_i[idx_k],C_vector+idx_k*5,datapoints,h_k+idx_k,weights,x);
	}
}

void h_l_gauss_integration_full_f(int ionizations_number,float T_e,float mu,float T_r,float *E_i,float *D_vector, int datapoints, float *h_l, float *weights, float *x)
{ int idx_l;
for  (idx_l=0;idx_l<ionizations_number;idx_l++)
	{
	h_l_gauss_integration_f(T_e,mu,E_i[idx_l],T_r,D_vector+idx_l*2,datapoints,h_l+idx_l,h_l+ionizations_number+idx_l,weights,x);
	}
}

int main(int argc, char *argv[])
{

int h_datapoints=32, ionizations_number=10000, excitations_number=10000, idx, h_block_mult=1;
if (argc>1){ionizations_number=atoi(argv[1]);}
if (argc>2){excitations_number=atoi(argv[2]);}
if (argc>3){h_block_mult=atoi(argv[3]);}
float *h_params, *E_i, *E_j, *B_vector, *C_vector, *D_vector, T_r;
float *h_j, *h_k, *h_l, *h_x, *h_w; float h_j2,h_k2,h_l2,h_le2;
FILE *INPUTFILE1, *INPUTFILE2;
clock_t h_start_t, h_end_t;
float h_total_t;

h_allocate_arrays_f(ionizations_number,excitations_number,h_datapoints,&h_params,&E_i,&E_j,&B_vector,&C_vector, &D_vector,&h_j, &h_k,&h_l,&h_w,&h_x);
gauss_integration_setup32_f(h_w,h_x);
h_params[0]=10.0f;	h_params[1]=3.0f;
T_r=300.0f;

if ((INPUTFILE1=fopen("Test_Ionization_Coeffs.txt", "r"))==NULL)
        {
        printf("Cannot open file! Error!\n");
        exit(2);
       	}

for(idx=0;idx<ionizations_number;idx++)
	{
	fscanf(INPUTFILE1,"%f %f %f %f %f %f %f %f", &E_i[idx], &C_vector[idx*5], &C_vector[idx*5+1], &C_vector[idx*5+2], &C_vector[idx*5+3], &C_vector[idx*5+4], &D_vector[idx*2], &D_vector[idx*2+1]);
	}

fclose(INPUTFILE1);

if ((INPUTFILE2=fopen("Test_Excitation_Coeffs.txt", "r"))==NULL)
        {
        printf("Cannot open file! Error!\n");
        exit(2);
       	}

for(idx=0;idx<excitations_number;idx++)
	{
	fscanf(INPUTFILE2,"%f %f %f %f %f", &E_j[idx], &B_vector[idx*4], &B_vector[idx*4+1], &B_vector[idx*4+2], &B_vector[idx*4+3]);
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

float *d_params, *d_B_vector, *d_C_vector, *d_D_vector, *d_E_j, *d_E_i, *d_j, *d_k, *d_k_part, *d_l, *d_x, *d_w;
d_setup_f(&d_params,&d_B_vector,&d_C_vector,&d_D_vector,&d_E_j,&d_E_i, &d_j, &d_k, &d_l, &d_x, &d_w,B_vector, C_vector, D_vector, E_j, E_i, T_r, h_x,h_w, ionizations_number,excitations_number, h_datapoints,streams, h_block_mult);
cudaMalloc((void *)d_k_part,sizeof(float)*h_datapoints*ionizations_number);

h_start_t=clock();
cudaEventRecord(start);
cudaEventRecord(start1,streams[0]);
cudaEventRecord(start2,streams[1]);

d_calculate_rates_fmem(d_params,d_B_vector, d_C_vector,d_D_vector, d_E_j, d_E_i, d_j, d_k, d_k_part, d_l, d_x,  d_w,h_params,h_j,h_k,h_l,h_w,h_x,T_r, ionizations_number,excitations_number,h_datapoints,streams, h_block_mult);

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
h_total_t=(float)(h_end_t-h_start_t)/CLOCKS_PER_SEC;
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


d_cleanup(d_params, d_B_vector, d_C_vector, d_E_j, d_E_i, d_j, d_k, d_k_part, d_l, d_x, d_w);
cudaFree(d_k_part);
exit(0);

}


