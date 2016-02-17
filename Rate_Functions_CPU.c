#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <lapacke.h>
#include <cblas.h>
#include <lapacke_mangling.h>
#include <lapacke_utils.h>
#include "Rate_Functions_CPU.h"

#define ACC_P 5
#define ACC_L 6503.0
#define ACC_IB1 6.5
#define ACC_IB2 15.0


//The prefix h_ corresponds to host (namely, CPU)

//Simple routine to find the sum of an array
double vector_sum(int length, double *vector)
{ int idx; double sum=0.0;
  for(idx=0;idx<length;idx++)
	{
	sum+=vector[idx];
	}
  return sum;
}

//Calculate electron density n_e, assuming quasineutrality
//Requires charge_vector, which contains the charge state of each element of N (the ion density vector)
double Get_n_e(int states_number, double *N, double *charge_vector)
{ double n_e; 
  n_e=cblas_ddot(states_number,N,1,charge_vector,1);
  return n_e;
}

//Fermi Energy/Temperature (they are synonymous, since k_B=1), namely the chemical potential at T=0
//Referred to as T_F throughout
double Fermi_Energy(double n)
{
	return 3.646450287910599E-15*pow(n,0.6666666666666667);
}

//Pade expansion of the chemical potential as a function of temperaure scaled to the Fermi energy as given in
//V.V.Karasiev et al, Computer Physics Communications 192, 114 (2015)
double Get_Chemical_Potential(double T, double T_F)
{ if(T==0.0)	{return T_F;}
  else
  {
	double u=T_F/T, top, bottom, u_pow;
	u_pow=u;
	top=1.5*log(u)-0.28468287047291913-0.6401645973717435*sqrt(u)*u*u;
	top+=0.017785515122132903*u_pow;
	u_pow*=u;
	top+=0.635477087603234*u_pow;
	bottom=1.0+0.04735469821665339*u_pow;
	u_pow*=u;
	top-=0.1330250650800089*u_pow;
	u_pow*=u;
	top-=0.0009706722152384041*u_pow;
	bottom+=0.40373166427559887*u_pow;
	u_pow*=u;
	top+=0.1307640895026443*u_pow;
	u_pow*=u;
	top+=0.008287996179592494*u_pow;
	bottom+=0.22608545408904693*u_pow;
	u_pow*=u;
	top+=0.20644821254230475*u_pow;
	u_pow*=u;
	bottom+=0.023573397905129*u_pow;
	u_pow*=u;
	top+=0.023573397905129*u_pow;
	return T*top/bottom;
  }
}

//Find heat capacity of Fermi gas, C_V using a Pade approximation
double Get_C_V(double Eta)	//Eta=mu/T
{double u, u_pow, top, bottom;
   if(Eta<2.0)
     {
	u=exp(Eta);
	top=4.32326386604283e4+8.55472308218786e4*u;		
	bottom=3.25218725353467e4+7.01022511904373e4*u;
	u_pow=u*u;
	top+=5.95275291210962e4*u_pow;
	bottom+=5.50859144223638e4*u_pow;
	u_pow*=u;
	top+=1.77294861572005e4*u_pow;
	bottom+=1.95942074576400e4*u_pow;
	u_pow*=u;
	top+=2.21876607796460e3*u_pow;
	bottom+=3.20803912586318e3*u_pow;
	u_pow*=u;
	top+=9.90562948053193e1*u_pow;
	bottom+=2.20853967067789e2*u_pow;
	u_pow*=u;
	top+=u_pow;
	bottom+=5.05580641737527e0*u_pow;
	u_pow*=u;
	bottom+=1.99507945223266e-2*u_pow;
	return u*top/bottom;	//Need to multiply by 1.5*T_e^(5/2)/T_F^(3/2) to get C_V
    }	
   else
     {
	u=1.0/(Eta*Eta);
	top=2.80452693148553e-13+8.60096863656367e-11*u;
	bottom=7.01131732871184e-13+2.10699282897576e-10*u;
	u_pow=u*u;
	top+=1.62974620742993e-8*u_pow;
	bottom+=3.94452010378723e-8*u_pow;
	u_pow*=u;
	top+=1.63598843752050e-6*u_pow;
	bottom+=3.84703231868724e-6*u_pow;
	u_pow*=u;
	top+=9.12915407846722e-5*u_pow;
	bottom+=2.04569943213216e-4*u_pow;
	u_pow*=u;
	top+=2.62988766922117e-3*u_pow;
	bottom+=5.31999109566385e-3*u_pow;
	u_pow*=u;
	top+=3.85682997219346e-2*u_pow;
	bottom+=6.39899717779153e-2*u_pow;
	u_pow*=u;
	top+=2.78383256609605e-1*u_pow;
	bottom+=3.14236143831882e-1*u_pow;
	u_pow*=u;
	top+=9.02250179334496e-1*u_pow;
	bottom+=4.70252591891375e-1*u_pow;
	u_pow*=u;
	top+=u_pow;
	bottom-=2.15540156936373e-2*u_pow;
	u_pow*=u;
	bottom+=2.34829436438087e-3*u_pow;
	return sqrt(Eta)*Eta*Eta*top/bottom;	//Need to multiply by 1.5*T_e^(5/2)/T_F^(3/2) to get C_V
    }
}

//Pade approximation of T(C_V), i.e. temperature as a function of heat capacity
//Used to find the temperature given the average electron energy
//For Maxwellian, C_V=1.5*T_e
double Invert_C_V(double u) 	//u=C_V/T_F
{  double u_pow, top, bottom;
   if(u>0.6)
     {
	top=-0.91174278812060261+4.6006033754292224*u;
	bottom=1.0-2.9841333233951111*u;
	u_pow=u*u;
	top-=10.223966412740976*u_pow;
	bottom+=6.5997384756821056*u_pow;
	u_pow*=u;
	top+=15.706434130616152*u_pow;
	bottom+=-2.0394444560416654*u_pow;
	u_pow*=u;
	top-=7.0042497246360398*u_pow;
	bottom+=-33.512500725339258*u_pow;
	u_pow*=u;
	top-=24.771955462425524*u_pow;
	bottom+=39.277208838856268*u_pow;
	u_pow*=u;
	top+=26.043226665533894*u_pow;
	bottom+=3.1012686268774381*u_pow;
	u_pow*=u;
	top+=2.0856967205125216*u_pow;
	bottom+=-0.00087411704884438578*u_pow;
	u_pow*=u;
	top-=0.0013111755732665787*u_pow;
	return top/bottom; //Returns T/T_F
     }
   else {return 0.0;}
}

//Evaluate the collision strength
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

//Integrand for inverse bremsstrahlung integral
double h_ib_int(double EGamma,double T_e,double mu,double T_r)
{	double exp_mu=exp(-mu/T_e);

	//This IF loop is required for stability (otherwise exp() function overflows)
	//The full formula tends to 0 for such large photon energies
	if (EGamma/T_e>5E2){return 0.0;}
	else {return (EGamma-T_e*log((exp(EGamma/T_e)*exp_mu+1.0)/(exp_mu+1.0)))/(exp(EGamma/T_r)-1.0);}
}

//Set up the nodes (x) and weights (w) of Gaussian quadrature, using the Golub Welsch algorithm
//coeffs and vectors are allocated arrays which are required only for solving the tridiagonal matrix equation
void gauss_integration_setup(int datapoints, double *weights, double *x,double *coeffs, double *vectors)
{  int idx; //int info;

   x[0]=0.0;
   for (idx=1;idx<datapoints;idx++)
	{
	x[idx]=0.0;
	coeffs[idx-1]=0.5/sqrt(1.0-1.0/(4.0*idx*idx));
	}	

   //dstev finds the eigenvalues and vectors of a symmetric 
   LAPACKE_dstev(LAPACK_ROW_MAJOR,'v', datapoints, x, coeffs, vectors, datapoints);

   for (idx=0;idx<datapoints;idx++)
	{
	x[idx]=0.5*(x[idx]+1.0);
	weights[idx]=vectors[idx]*vectors[idx];
	}
}

//Full collisional excitation calculation
void h_j_gauss_integration(double T_e,double mu,double E_j,double *B_vector, int datapoints, double *h_j_up, double *weights, double *x)
{	double integrand=0.0, E0, fermi, fermi_m, integ_temp;
	double region_difference=(20.0+(5.7+2.13*ACC_P)*T_e);	
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
	double region_difference=(30.0+(12.0+2.31*ACC_P)*T_e);
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

//Full inverse bremsstrahlung calculation
//Note that there is no equivalent GPU calculation
double h_ib_gauss_integration(double T_e, double n_e,double mu,double T_r, int states_number, double *charge_vector, double *N, int datapoints, double *weights, double *x)
{	double integrand=0.0, start=3.71075E-13*sqrt(n_e), end, EGamma; int idx;
	if (T_e>3.0)
		{end=(T_r/T_e)*(11.66+mu*0.066357/T_e);}
	else
		{end=(T_r/T_e)*(11.97+mu*0.0053961/T_e);}
	double region_difference=end-start;
   for(idx=0;idx<datapoints;idx++)
	{
		EGamma=x[idx]*region_difference+start;
		integrand+=weights[idx]*h_ib_int(EGamma,T_e,mu,T_r);
	}
	double sum;
   for(idx=0;idx<states_number;idx++)
	{
		sum+=charge_vector[idx]*charge_vector[idx]*N[idx];
	}
   return 2.29908947E9*sum*integrand*region_difference;
}

//CPU memory allocation
void h_allocate_arrays(int states_number, int ionizations_number, int excitations_number, int h_datapoints, double **h_params, double **charge_vector, double **E_i,double **E_j,double **A_vector,double **B_vector1, double **B_vector2, double **C_vector, double **D_vector, double **h_j, double **h_k, double **h_l, double **h_w, double **h_x, double **h_c, double **h_v, double **N, double **N_temp1, double **N_temp2, double **N_temp3, double **N_temp4, double **IntE_temp, double **R_1, double **R_2, int **excitations_indices, int **ionizations_indices)
{ int idx1, idx2;
*h_params=(double*)malloc(2*sizeof(double));
*E_i=(double*)malloc(ionizations_number*sizeof(double));
*E_j=(double*)malloc(excitations_number*sizeof(double));
*A_vector=(double*)malloc(excitations_number*sizeof(double));
*B_vector1=(double*)malloc(excitations_number*4*sizeof(double));
*C_vector=(double*)malloc(ionizations_number*5*sizeof(double));
*D_vector=(double*)malloc(ionizations_number*2*sizeof(double));
*h_j=(double*)malloc(excitations_number*sizeof(double));
*h_k=(double*)malloc(ionizations_number*sizeof(double));
*h_l=(double*)malloc(2*ionizations_number*sizeof(double));
*h_x=(double*)malloc(h_datapoints*sizeof(double));		
*h_w=(double*)malloc(h_datapoints*sizeof(double));
*h_c=(double*)malloc((h_datapoints-1)*sizeof(double));	
*h_v=(double*)malloc(h_datapoints*h_datapoints*sizeof(double));

*N=(double*)malloc(states_number*sizeof(double));
*charge_vector=(double*)malloc(states_number*sizeof(double));
*N_temp1=(double*)malloc(states_number*sizeof(double));
*N_temp2=(double*)malloc(states_number*sizeof(double));
*N_temp3=(double*)malloc(states_number*sizeof(double));
*N_temp4=(double*)malloc(states_number*sizeof(double));
*IntE_temp=(double*)malloc(states_number*sizeof(double));
*R_1=(double*)malloc(states_number*states_number*sizeof(double));
*R_2=(double*)malloc(states_number*states_number*sizeof(double));

for(idx1=0;idx1<states_number;idx1++)
	{
	for(idx2=0;idx2<states_number;idx2++) {R_1[0][idx1*states_number+idx2]=0.0;R_2[0][idx1*states_number+idx2]=0.0;}
	}

*excitations_indices=(int*)malloc(2*excitations_number*sizeof(int));
*ionizations_indices=(int*)malloc(2*ionizations_number*sizeof(int));
}

//Populate the relevant arrays with values specific to an atomic model 
void h_setup_atomic_model(int ionizations_number, int excitations_number,int *excitations_indices,int *ionizations_indices, double *charge_vector, double *E_i, double *E_j, double *A_vector, double *B_vector1, double *C_vector, double *D_vector)
{
int idx;

//Atomic model for aluminum, from 3+ to 6+

//The charge of each atomic level
charge_vector[0]=3.0; charge_vector[1]=3.0; charge_vector[2]=3.0; charge_vector[3]=3.0; charge_vector[4]=3.0; 
charge_vector[5]=4.0; charge_vector[6]=4.0; charge_vector[7]=4.0; charge_vector[8]=4.0; charge_vector[9]=4.0;
charge_vector[10]=5.0; charge_vector[11]=5.0; charge_vector[12]=5.0; charge_vector[13]=5.0;
charge_vector[14]=6.0;

//E_i = ionization energy
E_i[0]=1.20E+02;	E_i[1]=4.25752E+01;	E_i[2]=3.44987E+01;	E_i[3]=2.49360E+01;	E_i[4]=2.09260E+01;
E_i[5]=1.54E+02;	E_i[6]=1.09378E+02;	E_i[7]=5.80140E+01;	E_i[8]=4.89413E+01;	E_i[9]=3.78600E+01;
E_i[10]=1.91E+02;	E_i[11]=1.50557E+02;	E_i[12]=1.00046E+02;	E_i[13]=7.39200E+01;
//Indices used to populate the rate matrix
ionizations_indices[0]=75;ionizations_indices[14]=5;
ionizations_indices[1]=76;ionizations_indices[15]=20;
ionizations_indices[2]=77;ionizations_indices[16]=35;
ionizations_indices[3]=78;ionizations_indices[17]=50;
ionizations_indices[4]=79;ionizations_indices[18]=65;
ionizations_indices[5]=155;ionizations_indices[19]=85;
ionizations_indices[6]=156;ionizations_indices[20]=100;
ionizations_indices[7]=157;ionizations_indices[21]=115;
ionizations_indices[8]=158;ionizations_indices[22]=130;
ionizations_indices[9]=159;ionizations_indices[23]=145;
ionizations_indices[10]=220;ionizations_indices[24]=164;
ionizations_indices[11]=221;ionizations_indices[25]=179;
ionizations_indices[12]=222;ionizations_indices[26]=194;
ionizations_indices[13]=223;ionizations_indices[27]=209;

//E_j = excitation energy
E_j[0]=7.74E+01;	E_j[1]=8.08E+00;	E_j[2]=9.89E+00;	E_j[3]=1.49E+01;	E_j[4]=9.51E+01;	E_j[5]=9.91E+01;
E_j[6]=4.43E+01;	E_j[7]=9.57E+01;	E_j[8]=9.07E+00;	E_j[9]=1.16E+02;	E_j[10]=1.11E+01;
E_j[11]=3.99E+01;	E_j[12]=5.05E+01;	E_j[13]=1.17E+02;
//Indices used to populate the rate matrix
excitations_indices[0]=15;excitations_indices[14]=1;
excitations_indices[1]=31;excitations_indices[15]=17;
excitations_indices[2]=47;excitations_indices[16]=33;
excitations_indices[3]=62;excitations_indices[17]=34;
excitations_indices[4]=45;excitations_indices[18]=3;
excitations_indices[5]=60;excitations_indices[19]=4;
excitations_indices[6]=95;excitations_indices[20]=81;
excitations_indices[7]=110;excitations_indices[21]=82;
excitations_indices[8]=127;excitations_indices[22]=113;
excitations_indices[9]=140;excitations_indices[23]=84;
excitations_indices[10]=143;excitations_indices[24]=129;
excitations_indices[11]=175;excitations_indices[25]=161;
excitations_indices[12]=191;excitations_indices[26]=177;
excitations_indices[13]=205;excitations_indices[27]=163;

//Coefficients for excitation cross section
B_vector1[0]=1.393000E-01;	B_vector1[1]=-7.979000E-02;	B_vector1[2]=1.057000E-01;	B_vector1[3]=2.630000E-03;
B_vector1[4]=7.078345E+01;	B_vector1[5]=3.952529E+01;	B_vector1[6]=9.925136E+01;	B_vector1[7]=-2.871901E+01;
B_vector1[8]=1.687015E+02;	B_vector1[9]=5.317823E+01;	B_vector1[10]=2.015390E+02;	B_vector1[11]=-4.346805E+01;
B_vector1[12]=1.582396E+01;	B_vector1[13]=-1.274900E+01;	B_vector1[14]=2.527637E+01;	B_vector1[15]=-6.928126E+00;
B_vector1[16]=5.332000E-01;	B_vector1[17]=-2.181000E-01;	B_vector1[18]=4.292000E-01;	B_vector1[19]=-8.999000E-02;
B_vector1[20]=2.345149E-02;	B_vector1[21]=-7.918600E-03;	B_vector1[22]=8.853680E-03;	B_vector1[23]=1.186270E-02;
B_vector1[24]=1.012000E+00;	B_vector1[25]=9.015000E-01;	B_vector1[26]=1.081000E+00;	B_vector1[27]=-2.239000E-01;
B_vector1[28]=6.215553E-01;	B_vector1[29]=-3.925198E-01;	B_vector1[30]=5.820666E-01;	B_vector1[31]=3.439001E-02;
B_vector1[32]=1.413465E+02;	B_vector1[33]=8.125362E+01;	B_vector1[34]=2.686247E+02;	B_vector1[35]=-9.678118E+01;
B_vector1[36]=3.901087E+00;	B_vector1[37]=-9.877649E-01;	B_vector1[38]=2.454189E+00;	B_vector1[39]=7.424383E-01;
B_vector1[40]=2.997265E+02;	B_vector1[41]=1.240335E+02;	B_vector1[42]=5.146290E+02;	B_vector1[43]=-1.742407E+02;
B_vector1[44]=2.857600E+00;	B_vector1[45]=2.633000E+00;	B_vector1[46]=3.919000E+00;	B_vector1[47]=-9.664000E-01;
B_vector1[48]=9.041080E-01;	B_vector1[49]=8.940613E-01;	B_vector1[50]=7.836745E-01;	B_vector1[51]=-2.640410E-02;
B_vector1[52]=9.889809E-01;	B_vector1[53]=-7.141192E-01;	B_vector1[54]=1.130646E+00;	B_vector1[55]=-8.393918E-02;

//Coefficients for differential cross section
C_vector[0]=3.71000E+00*E_i[0];		C_vector[1]=-2.36000E+00*0;	C_vector[2]=2.35000E+00;	C_vector[3]=-6.60000E-01;	C_vector[4]=0.00000E+00;
C_vector[5]=3.71000E+00*E_i[1];		C_vector[6]=-2.36000E+00*0;	C_vector[7]=2.35000E+00;	C_vector[8]=-6.60000E-01;	C_vector[9]=0.00000E+00;
C_vector[10]=3.71000E+00*E_i[2];	C_vector[11]=-2.36000E+00*0;	C_vector[12]=2.35000E+00;	C_vector[13]=-6.60000E-01;	C_vector[14]=0.00000E+00;
C_vector[15]=3.71000E+00*E_i[3];	C_vector[16]=-2.36000E+00*0;	C_vector[17]=2.35000E+00;	C_vector[18]=-6.60000E-01;	C_vector[19]=0.00000E+00;
C_vector[20]=3.71000E+00*E_i[4];	C_vector[21]=-2.36000E+00*0;	C_vector[22]=2.35000E+00;	C_vector[23]=-6.60000E-01;	C_vector[24]=0.00000E+00;
C_vector[25]=2.79400E+00*E_i[5];	C_vector[26]=4.69000E-01*0;	C_vector[27]=-1.29400E+01;	C_vector[28]=2.62600E+01;	C_vector[29]=-1.34300E+01;
C_vector[30]=2.79400E+00*E_i[6];	C_vector[31]=4.69000E-01*0;	C_vector[32]=-1.29400E+01;	C_vector[33]=2.62600E+01;	C_vector[34]=-1.34300E+01;
C_vector[35]=2.79400E+00*E_i[7];	C_vector[36]=4.69000E-01*0;	C_vector[37]=-1.29400E+01;	C_vector[38]=2.62600E+01;	C_vector[39]=-1.34300E+01;
C_vector[40]=2.79400E+00*E_i[8];	C_vector[41]=4.69000E-01*0;	C_vector[42]=-1.29400E+01;	C_vector[43]=2.62600E+01;	C_vector[44]=-1.34300E+01;
C_vector[45]=2.79400E+00*E_i[9];	C_vector[46]=4.69000E-01*0;	C_vector[47]=-1.29400E+01;	C_vector[48]=2.62600E+01;	C_vector[49]=-1.34300E+01;
C_vector[50]=2.41900E+00*E_i[10];	C_vector[51]=-1.32000E+00*0;	C_vector[52]=1.70000E+00;	C_vector[53]=0.00000E+00;	C_vector[54]=0.00000E+00;
C_vector[55]=2.41900E+00*E_i[11];	C_vector[56]=-1.32000E+00*0;	C_vector[57]=1.70000E+00;	C_vector[58]=0.00000E+00;	C_vector[59]=0.00000E+00;
C_vector[60]=2.41900E+00*E_i[12];	C_vector[61]=-1.32000E+00*0;	C_vector[62]=1.70000E+00;	C_vector[63]=0.00000E+00;	C_vector[64]=0.00000E+00;
C_vector[65]=2.41900E+00*E_i[13];	C_vector[66]=-1.32000E+00*0;	C_vector[67]=1.70000E+00;	C_vector[68]=0.00000E+00;	C_vector[69]=0.00000E+00;

//Coefficients for the photoionization cross section
D_vector[0]=5.682286E-18*E_i[0]*E_i[0];		D_vector[1]=3.941713E-19*E_i[0]*E_i[0]*E_i[0];
D_vector[2]=5.682286E-18*E_i[1]*E_i[1];		D_vector[3]=3.941713E-19*E_i[1]*E_i[1]*E_i[1];
D_vector[4]=5.682286E-18*E_i[2]*E_i[2];		D_vector[5]=3.941713E-19*E_i[2]*E_i[2]*E_i[2];
D_vector[6]=5.682286E-18*E_i[3]*E_i[3];		D_vector[7]=3.941713E-19*E_i[3]*E_i[3]*E_i[3];
D_vector[8]=5.682286E-18*E_i[4]*E_i[4];		D_vector[9]=3.941713E-19*E_i[4]*E_i[4]*E_i[4];
D_vector[10]=1.977137E-18*E_i[5]*E_i[5];	D_vector[11]=1.844077E-18*E_i[5]*E_i[5]*E_i[5];
D_vector[12]=1.977137E-18*E_i[6]*E_i[6];	D_vector[13]=1.844077E-18*E_i[6]*E_i[6]*E_i[6];
D_vector[14]=1.977137E-18*E_i[7]*E_i[7];	D_vector[15]=1.844077E-18*E_i[7]*E_i[7]*E_i[7];
D_vector[16]=1.977137E-18*E_i[8]*E_i[8];	D_vector[17]=1.844077E-18*E_i[8]*E_i[8]*E_i[8];
D_vector[18]=1.977137E-18*E_i[9]*E_i[9];	D_vector[19]=1.844077E-18*E_i[9]*E_i[9]*E_i[9];
D_vector[20]=7.897529E-19*E_i[10]*E_i[10];	D_vector[21]=1.390636E-18*E_i[10]*E_i[10]*E_i[10];
D_vector[22]=7.897529E-19*E_i[11]*E_i[11];	D_vector[23]=1.390636E-18*E_i[11]*E_i[11]*E_i[11];
D_vector[24]=7.897529E-19*E_i[12]*E_i[12];	D_vector[25]=1.390636E-18*E_i[12]*E_i[12]*E_i[12];
D_vector[26]=7.897529E-19*E_i[13]*E_i[13];	D_vector[27]=1.390636E-18*E_i[13]*E_i[13]*E_i[13];

 for(idx=0;idx<excitations_number*4;idx++) {B_vector1[idx]*=4.831E14;}
 for(idx=0;idx<ionizations_number*5;idx++) {C_vector[idx]*=4.036E16;}
 for(idx=0;idx<ionizations_number*2;idx++) {D_vector[idx]*=3.94286E+23;}
}

//Populate the rate matrices
//R_1 is the standard rate matrix for the change in ion populations
//R_2 is an energy matrix - this corresponds to the change in free electron kinetic energy because of transfer to potential energy
void h_create_rate_matrices(int states_number, int ionizations_number, int excitations_number, double T_e, double n_e, double mu, double *R_1, double *R_2, double *j, double *k, double *l, int *excitations_indices, int *ionizations_indices, double *E_i, double *E_j)
{	int idx, idx2; double sum, down_rate;
	for(idx=0;idx<excitations_number;idx++)
		{
		R_1[excitations_indices[idx]]=j[idx];
		R_2[excitations_indices[idx]]=-j[idx]*E_j[idx];
		down_rate=j[idx]*exp(E_j[idx]/T_e);	//Using the Boltzmann relation - detailed balance
		R_1[excitations_indices[idx+excitations_number]]=down_rate;
		R_2[excitations_indices[idx+excitations_number]]=down_rate*E_j[idx];
		}

	for(idx=0;idx<ionizations_number;idx++)
		{
		R_1[ionizations_indices[idx]]=k[idx]+l[idx];
		R_2[ionizations_indices[idx]]=-k[idx]*E_i[idx]+l[idx+ionizations_number];
		down_rate=k[idx]*exp((mu+E_i[idx])/T_e)/n_e; 	//Detailed balance for degenerate three body recombination
		R_1[ionizations_indices[idx+ionizations_number]]=down_rate;
		R_2[ionizations_indices[idx+ionizations_number]]=down_rate*E_i[idx];
		}

	//Diagonal of the rate matrix R_1 is the negative sum of the rest of the corresponding column, to ensure conservation of atom number
	for(idx=0;idx<states_number;idx++)
		{
		R_1[idx*(1+states_number)]=0.0;
		sum=0.0;
		for(idx2=0;idx2<states_number;idx2++)
			{
			sum+=R_1[idx2*states_number+idx];
			}
		R_1[idx*(1+states_number)]=-sum;
		}
}

//Generic Runge-Kutta 4 solver
//Solving for N (vector of ion densities, determines electron density n_e by quasineturality) and Internal_Energy (determines the temperature)
//RK-4 method, N_t[n+1] = N_t[n]+Delta_t/6*(k_1+2k_2+2k_3+k_4)
//At each timestep: 
// (1) Calculate plasma properties from previous timestep: n_e, T_F, T_e, mu
// (2) Calculate atomic rates
// (3) Populate rate matrices
// (4) Calculate the temporary increment k_1 or k_2 etc
void h_solve_RK4(int states_number, int ionizations_number, int excitations_number, double delta, double *charge_vector, double *N, double *N_temp1, double *N_temp2, double *N_temp3, double *N_temp4, double *IntE_temp, double *n_e, double *T_e, double *T_F, double *Internal_Energy,int h_datapoints, double *h_w, double *h_x, double *h_j, double *h_k, double *h_l, double T_r, int *excitations_indices, int *ionizations_indices, double *E_i, double *E_j,double *A_vector, double *B_vector1, double *C_vector, double *D_vector, double *R_1, double *R_2)
{	
	//Declare variables, copy current values to temporary arrays
	double n_e_temp, T_F_temp, T_e_temp, delta_2=0.5*delta, Int_energy_temp1, Int_energy_temp2, Int_energy_temp3, Int_energy_temp4, mu, ib_E;
	cblas_dcopy(states_number,N,1,N_temp1,1);
	cblas_dcopy(states_number,N,1,N_temp2,1);
	cblas_dcopy(states_number,N,1,N_temp3,1);
	cblas_dcopy(states_number,N,1,N_temp4,1);

	//Coefficients 1
	mu=Get_Chemical_Potential(*T_e,*T_F);
	h_j_gauss_integration_full(excitations_number,*T_e,mu,E_j,B_vector1,h_datapoints,h_j,h_w,h_x);
	h_k_gauss_integration_full(ionizations_number,*T_e,mu,E_i,C_vector,h_datapoints,h_k,h_w,h_x);
	h_l_gauss_integration_full(ionizations_number,*T_e,mu,T_r,E_i,D_vector,h_datapoints,h_l,h_w,h_x);
	ib_E=h_ib_gauss_integration(*T_e,*n_e,mu,T_r,states_number,charge_vector,N,h_datapoints,h_w,h_x);
	h_create_rate_matrices(states_number,ionizations_number,excitations_number,*T_e,*n_e,mu,R_1,R_2,h_j,h_k,h_l,excitations_indices,ionizations_indices,E_i,E_j);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_1,states_number,N,1,1.0,N_temp1,1);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_2,states_number,N,1,0.0,IntE_temp,1);
	Int_energy_temp1=vector_sum(states_number,IntE_temp)+*Internal_Energy+ib_E*delta_2;

	//Coefficients 2
	n_e_temp=Get_n_e(states_number,N_temp1,charge_vector);
	T_F_temp=Fermi_Energy(n_e_temp);
	T_e_temp=T_F_temp*Invert_C_V(Int_energy_temp1/(T_F_temp*n_e_temp));
	mu=Get_Chemical_Potential(T_e_temp,T_F_temp);
	h_j_gauss_integration_full(excitations_number,T_e_temp,mu,E_j,B_vector1,h_datapoints,h_j,h_w,h_x);
	h_k_gauss_integration_full(ionizations_number,T_e_temp,mu,E_i,C_vector,h_datapoints,h_k,h_w,h_x);
	h_l_gauss_integration_full(ionizations_number,T_e_temp,mu,T_r,E_i,D_vector,h_datapoints,h_l,h_w,h_x);
	ib_E=h_ib_gauss_integration(T_e_temp,n_e_temp,mu,T_r,states_number,charge_vector,N_temp1,h_datapoints,h_w,h_x);
	h_create_rate_matrices(states_number,ionizations_number,excitations_number,T_e_temp,n_e_temp,mu,R_1,R_2,h_j,h_k,h_l,excitations_indices,ionizations_indices,E_i,E_j);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_1,states_number,N_temp1,1,1.0,N_temp2,1);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_2,states_number,N_temp1,1,0.0,IntE_temp,1);
	Int_energy_temp2=vector_sum(states_number,IntE_temp)+*Internal_Energy+ib_E*delta_2;

	//Coefficients 3
	n_e_temp=Get_n_e(states_number,N_temp2,charge_vector);
	T_F_temp=Fermi_Energy(n_e_temp);
	T_e_temp=T_F_temp*Invert_C_V(Int_energy_temp2/(T_F_temp*n_e_temp));
	mu=Get_Chemical_Potential(T_e_temp,T_F_temp);
	h_j_gauss_integration_full(excitations_number,T_e_temp,mu,E_j,B_vector1,h_datapoints,h_j,h_w,h_x);
	h_k_gauss_integration_full(ionizations_number,T_e_temp,mu,E_i,C_vector,h_datapoints,h_k,h_w,h_x);
	h_l_gauss_integration_full(ionizations_number,T_e_temp,mu,T_r,E_i,D_vector,h_datapoints,h_l,h_w,h_x);
	ib_E=h_ib_gauss_integration(T_e_temp,n_e_temp,mu,T_r,states_number,charge_vector,N_temp2,h_datapoints,h_w,h_x);
	h_create_rate_matrices(states_number,ionizations_number,excitations_number,T_e_temp,n_e_temp,mu,R_1,R_2,h_j,h_k,h_l,excitations_indices,ionizations_indices,E_i,E_j);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta,R_1,states_number,N_temp2,1,1.0,N_temp3,1);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta,R_2,states_number,N_temp2,1,0.0,IntE_temp,1);
	Int_energy_temp3=vector_sum(states_number,IntE_temp)+*Internal_Energy+ib_E*delta;

	//Coefficients 4
	n_e_temp=Get_n_e(states_number,N_temp3,charge_vector);
	T_F_temp=Fermi_Energy(n_e_temp);
	T_e_temp=T_F_temp*Invert_C_V(Int_energy_temp3/(T_F_temp*n_e_temp));
	mu=Get_Chemical_Potential(T_e_temp,T_F_temp);
	h_j_gauss_integration_full(excitations_number,T_e_temp,mu,E_j,B_vector1,h_datapoints,h_j,h_w,h_x);
	h_k_gauss_integration_full(ionizations_number,T_e_temp,mu,E_i,C_vector,h_datapoints,h_k,h_w,h_x);
	h_l_gauss_integration_full(ionizations_number,T_e_temp,mu,T_r,E_i,D_vector,h_datapoints,h_l,h_w,h_x);
	ib_E=h_ib_gauss_integration(T_e_temp,n_e_temp,mu,T_r,states_number,charge_vector,N_temp3,h_datapoints,h_w,h_x);
	h_create_rate_matrices(states_number,ionizations_number,excitations_number,T_e_temp,n_e_temp,mu,R_1,R_2,h_j,h_k,h_l,excitations_indices,ionizations_indices,E_i,E_j);
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

///////////////////////////////////////////
//Calculation of Maxwellian rates and creation of appropriate rate matrices
///////////////////////////////////////////

//First order exponential integral function; 100 max iterations
//Required for ionization, excitation calculations
double exp_int(double x)
{
double a,b,c,d,del,fact=1,h,expint=0;
int i;
if(x==0.0)
	{return 0.0;} //Ordinarily E_1(0)=inf, but this is done for stability
else if(x>1.0)       //Use convergent fraction for large argument
	{b=x+1;c=1E30;d=1/b;h=d; for(i=1;i<100;i++){a=-i*i;b=b+2;d=1/(a*d+b);c=b+a/c;del=c*d;h=h*del; if(fabs(del-1)<1E-7){expint=h*exp(-x);}}}
else 		     //Use convergent Taylor series for small argument
	{expint=-log(x)-0.5772156649; for(i=1;i<100;i++){fact=-fact*x/i; expint=expint-fact/i;}} 
return expint;
}

//Following are the rate coefficients obtained by analytically integrating over the Maxwell-Boltzmann distribution
double j_up1_maxwellian(double T_e,double E_j,double *B_vector)
{
	double E_prime=E_j/T_e;
	double E_prime2=E_prime*E_prime;
	return 1.658041813289174e-22/sqrt(T_e)*(exp(-E_prime)*(B_vector[1]+B_vector[3]*E_prime+B_vector[4]*0.5*(E_prime-E_prime2))+exp_int(E_prime)*(B_vector[0]+B_vector[2]*E_prime-B_vector[3]*E_prime2+B_vector[4]*E_prime2*E_prime*0.5));
}

double j_down_maxwellian(double j_up,double T_e,double E_j)
{
	return exp(E_j/T_e)*j_up;
}

double k_up_maxwellian(double T_e,double E_i,double *C_vector)
{	
	double t=sqrt(T_e);
	double G=exp_int(E_i/T_e)/T_e;
	double F=exp(-E_i/T_e)/E_i;
	double E2=E_i*E_i, E3=E2*E_i;
	double T2=T_e*T_e, T3=T2*T_e;
	return 1.659316E-22*(t*G*C_vector[0]/E2+(F-G)*C_vector[1]/t+((T_e+E_i)*F-(E_i+2.0*T_e)*G)*C_vector[2]/(t*T_e)+((E2+5.0*E_i*T_e+2.0*T2)*F-(E2+6.0*E_i*T_e+6.0*T2)*G)*C_vector[3]/(2.0*T2*t)+((E3+11.0*E2*T_e+26.0*E_i*T2+6.0*T3)*F-(E3+12.0*E2*T_e+36.0*E_i*T2+24.0*T3)*G)*C_vector[4]/(6.0*T3*t));
}

double k_down_maxwellian(double k_up,double T_e,double E_i)
{
	return exp(E_i/T_e)*k_up/(6.061E21*sqrt(T_e)*T_e);
}

//The following are numerical integrals over the Maxwell-Boltzmann multiplying the Bose-Einstein 
void l_up_maxwellian(double T_e,double E_i,double T_r,double *D_vector, int datapoints, double *h_l, double *h_le, double *weights, double *x)
{	double integrand0=0.0, integrand1=0.0, EGamma, EGammaPrime, fermi_m, integ_temp;
	double region_difference=ACC_L;
	int idx;

   for(idx=0;idx<datapoints;idx++)
	{
		EGammaPrime=x[idx]*region_difference;
		EGamma=EGammaPrime+E_i;
		integ_temp=h_l_int(EGamma,E_i,T_r,D_vector)*weights[idx];
		integrand0+=integ_temp;
		integrand1+=integ_temp*EGammaPrime;
	}
  *h_l=integrand0*region_difference;
  *h_le=integrand1*region_difference;
	
}

void j_maxwellian_full(int excitations_number,double T_e,double *E_j,double *B_vector, double *h_j)
{ int idx_j;

  for (idx_j=0;idx_j<excitations_number;idx_j++)
	{
	h_j[idx_j]=j_up1_maxwellian(T_e,E_j[idx_j],B_vector+idx_j*4);
	}
}

void k_maxwellian_full(int ionizations_number,double T_e,double *E_i,double *C_vector, double *h_k)
{ int idx_k;

for  (idx_k=0;idx_k<ionizations_number;idx_k++)
	{
	h_k[idx_k]=k_up_maxwellian(T_e,E_i[idx_k],C_vector+idx_k*5);
	}
}


void l_maxwellian_full(int ionizations_number,double T_e,double T_r,double *E_i,double *D_vector, int datapoints, double *h_l, double *weights, double *x)
{ int idx_l;
for  (idx_l=0;idx_l<ionizations_number;idx_l++)
	{
	l_up_maxwellian(T_e,E_i[idx_l],T_r,D_vector+idx_l*2,datapoints,h_l+idx_l,h_l+ionizations_number+idx_l,weights,x);
	}
}
double ib_maxwellian(double T_e, double n_e,double T_r, int states_number, double *charge_vector, double *N, int datapoints, double *weights, double *x)
{	double integrand=0.0, start=3.71075E-13*sqrt(n_e), end, EGamma; int idx;
	end=(T_r/T_e)*11.66;
	double region_difference=end-start;
   for(idx=0;idx<datapoints;idx++)
	{
		EGamma=x[idx]*region_difference+start;
		integrand+=weights[idx]*(1.0-exp(-EGamma/T_e))/(exp(EGamma/T_r)-1.0);
	}
	double sum;
   for(idx=0;idx<states_number;idx++)
	{
		sum+=charge_vector[idx]*charge_vector[idx]*N[idx];
	}
   return 3.793246E-13*sum*n_e*integrand*region_difference/sqrt(T_e);
}

////////////////////////////////
//The following two functions are equivalent to their Fermi-Dirac counterparts, see above
void create_rate_matrices_maxwellian(int states_number, int ionizations_number, int excitations_number, double T_e, double n_e, double *R_1, double *R_2, double *j, double *k, double *l, int *excitations_indices, int *ionizations_indices, double *E_i, double *E_j)
{	int idx, idx2; double sum, up_rate, down_rate;
	for(idx=0;idx<excitations_number;idx++)
		{
		up_rate=j[idx]*n_e;
		R_1[excitations_indices[idx]]=up_rate;
		R_2[excitations_indices[idx]]=-up_rate*E_j[idx];
		down_rate=j_down_maxwellian(up_rate,T_e,E_j[idx]);
		R_1[excitations_indices[idx+excitations_number]]=down_rate;
		R_2[excitations_indices[idx+excitations_number]]=down_rate*E_j[idx];
		}
	
	for(idx=0;idx<ionizations_number;idx++)
		{
		up_rate=k[idx]*n_e;
		R_1[ionizations_indices[idx]]=up_rate+l[idx];
		R_2[ionizations_indices[idx]]=-up_rate*E_i[idx]+l[idx+ionizations_number];
		down_rate=k_down_maxwellian(up_rate,T_e,E_i[idx])*n_e;
		R_1[ionizations_indices[idx+ionizations_number]]=down_rate;
		R_2[ionizations_indices[idx+ionizations_number]]=down_rate*E_i[idx];
		}

	for(idx=0;idx<states_number;idx++)
		{
		R_1[idx*(1+states_number)]=0.0;
		sum=0.0;
		for(idx2=0;idx2<states_number;idx2++)
			{
			sum+=R_1[idx2*states_number+idx];
			}
		R_1[idx*(1+states_number)]=-sum;
		}

}

//RK-4 solver for Maxwell-Boltzmann statistics
//Uses analytic integrals above and the usual heat capacity C_V=1.5*T_e
void solve_RK4_maxwellian(int states_number, int ionizations_number, int excitations_number, double delta, double *charge_vector, double *N, double *N_temp1, double *N_temp2, double *N_temp3, double *N_temp4, double *IntE_temp, double *n_e, double *T_e, double *Internal_Energy,int h_datapoints, double *h_w, double *h_x, double *h_j, double *h_k, double *h_l, double T_r, int *excitations_indices, int *ionizations_indices, double *E_i, double *E_j,double *A_vector, double *B_vector1, double *C_vector, double *D_vector, double *R_1, double *R_2)
{	
	//Copy to temporary arrays
	double n_e_temp, T_e_temp, delta_2=0.5*delta, Int_energy_temp1, Int_energy_temp2, Int_energy_temp3, Int_energy_temp4, ib_E;
	cblas_dcopy(states_number,N,1,N_temp1,1);
	cblas_dcopy(states_number,N,1,N_temp2,1);
	cblas_dcopy(states_number,N,1,N_temp3,1);
	cblas_dcopy(states_number,N,1,N_temp4,1);

	//Coefficients 1
	j_maxwellian_full(excitations_number,*T_e,E_j,B_vector1,h_j);
	k_maxwellian_full(ionizations_number,*T_e,E_i,C_vector,h_k);
	l_maxwellian_full(ionizations_number,*T_e,T_r,E_i,D_vector,h_datapoints,h_l,h_w,h_x);
	ib_E=ib_maxwellian(*T_e,*n_e,T_r,states_number,charge_vector,N,h_datapoints,h_w,h_x);
	create_rate_matrices_maxwellian(states_number,ionizations_number,excitations_number,*T_e,*n_e,R_1,R_2,h_j,h_k,h_l,excitations_indices,ionizations_indices,E_i,E_j);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_1,states_number,N,1,1.0,N_temp1,1);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_2,states_number,N,1,0.0,IntE_temp,1);
	Int_energy_temp1=vector_sum(states_number,IntE_temp)+*Internal_Energy+ib_E*delta_2;

	//Coefficients 2
	n_e_temp=Get_n_e(states_number,N_temp1,charge_vector);
	T_e_temp=0.6666666666666667*Int_energy_temp1/n_e_temp;
	j_maxwellian_full(excitations_number,T_e_temp,E_j,B_vector1,h_j);
	k_maxwellian_full(ionizations_number,T_e_temp,E_i,C_vector,h_k);
	l_maxwellian_full(ionizations_number,T_e_temp,T_r,E_i,D_vector,h_datapoints,h_l,h_w,h_x);
	ib_E=ib_maxwellian(T_e_temp,n_e_temp,T_r,states_number,charge_vector,N_temp1,h_datapoints,h_w,h_x);
	create_rate_matrices_maxwellian(states_number,ionizations_number,excitations_number,T_e_temp,n_e_temp,R_1,R_2,h_j,h_k,h_l,excitations_indices,ionizations_indices,E_i,E_j);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_1,states_number,N_temp1,1,1.0,N_temp2,1);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta_2,R_2,states_number,N_temp1,1,0.0,IntE_temp,1);
	Int_energy_temp2=vector_sum(states_number,IntE_temp)+*Internal_Energy+ib_E*delta_2;

	//Coefficients 3
	n_e_temp=Get_n_e(states_number,N_temp2,charge_vector);
	T_e_temp=0.6666666666666667*Int_energy_temp2/n_e_temp;
	j_maxwellian_full(excitations_number,T_e_temp,E_j,B_vector1,h_j);
	k_maxwellian_full(ionizations_number,T_e_temp,E_i,C_vector,h_k);
	l_maxwellian_full(ionizations_number,T_e_temp,T_r,E_i,D_vector,h_datapoints,h_l,h_w,h_x);
	ib_E=ib_maxwellian(T_e_temp,n_e_temp,T_r,states_number,charge_vector,N_temp2,h_datapoints,h_w,h_x);
	create_rate_matrices_maxwellian(states_number,ionizations_number,excitations_number,T_e_temp,n_e_temp,R_1,R_2,h_j,h_k,h_l,excitations_indices,ionizations_indices,E_i,E_j);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta,R_1,states_number,N_temp2,1,1.0,N_temp3,1);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,states_number,states_number,delta,R_2,states_number,N_temp2,1,0.0,IntE_temp,1);
	Int_energy_temp3=vector_sum(states_number,IntE_temp)+*Internal_Energy+ib_E*delta;

	//Coefficients 4
	n_e_temp=Get_n_e(states_number,N_temp3,charge_vector);
	T_e_temp=0.6666666666666667*Int_energy_temp3/n_e_temp;
	j_maxwellian_full(excitations_number,T_e_temp,E_j,B_vector1,h_j);
	k_maxwellian_full(ionizations_number,T_e_temp,E_i,C_vector,h_k);
	l_maxwellian_full(ionizations_number,T_e_temp,T_r,E_i,D_vector,h_datapoints,h_l,h_w,h_x);
	ib_E=ib_maxwellian(T_e_temp,n_e_temp,T_r,states_number,charge_vector,N_temp3,h_datapoints,h_w,h_x);
	create_rate_matrices_maxwellian(states_number,ionizations_number,excitations_number,T_e_temp,n_e_temp,R_1,R_2,h_j,h_k,h_l,excitations_indices,ionizations_indices,E_i,E_j);
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
	*T_e=0.6666666666666667**Internal_Energy/ *n_e;
}
//////////////////////////////////////////
