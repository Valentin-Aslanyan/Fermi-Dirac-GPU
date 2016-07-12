#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>



int main()
{
int idx, ionizations_number=14, excitations_number=14;
int new_ion_levs=100000, new_exc_levels=100000;
int current_ion_lev=0, current_exc_lev=0;
FILE *OUTPUTFILE1, *OUTPUTFILE2;

srand(time(NULL));

double *E_i, *E_j, *B_vector, *C_vector, *D_vector;
double E_i_new, E_j_new, B_vector1, B_vector2, B_vector3, B_vector4, C_vector1, C_vector2, C_vector3, C_vector4, C_vector5, D_vector1, D_vector2;

E_i=(double*)malloc(ionizations_number*sizeof(double));
E_j=(double*)malloc(excitations_number*sizeof(double));
B_vector=(double*)malloc(excitations_number*4*sizeof(double));
C_vector=(double*)malloc(ionizations_number*5*sizeof(double));
D_vector=(double*)malloc(ionizations_number*2*sizeof(double));

//E_i = ionization energy
E_i[0]=1.20E+02;	E_i[1]=4.25752E+01;	E_i[2]=3.44987E+01;	E_i[3]=2.49360E+01;	E_i[4]=2.09260E+01;
E_i[5]=1.54E+02;	E_i[6]=1.09378E+02;	E_i[7]=5.80140E+01;	E_i[8]=4.89413E+01;	E_i[9]=3.78600E+01;
E_i[10]=1.91E+02;	E_i[11]=1.50557E+02;	E_i[12]=1.00046E+02;	E_i[13]=7.39200E+01;


//E_j = excitation energy
E_j[0]=7.74E+01;	E_j[1]=8.08E+00;	E_j[2]=9.89E+00;	E_j[3]=1.49E+01;	E_j[4]=9.51E+01;	E_j[5]=9.91E+01;
E_j[6]=4.43E+01;	E_j[7]=9.57E+01;	E_j[8]=9.07E+00;	E_j[9]=1.16E+02;	E_j[10]=1.11E+01;
E_j[11]=3.99E+01;	E_j[12]=5.05E+01;	E_j[13]=1.17E+02;


//Coefficients for excitation cross section
B_vector[0]=1.393000E-01;	B_vector[1]=-7.979000E-02;	B_vector[2]=1.057000E-01;	B_vector[3]=2.630000E-03;
B_vector[4]=7.078345E+01;	B_vector[5]=3.952529E+01;	B_vector[6]=9.925136E+01;	B_vector[7]=-2.871901E+01;
B_vector[8]=1.687015E+02;	B_vector[9]=5.317823E+01;	B_vector[10]=2.015390E+02;	B_vector[11]=-4.346805E+01;
B_vector[12]=1.582396E+01;	B_vector[13]=-1.274900E+01;	B_vector[14]=2.527637E+01;	B_vector[15]=-6.928126E+00;
B_vector[16]=5.332000E-01;	B_vector[17]=-2.181000E-01;	B_vector[18]=4.292000E-01;	B_vector[19]=-8.999000E-02;
B_vector[20]=2.345149E-02;	B_vector[21]=-7.918600E-03;	B_vector[22]=8.853680E-03;	B_vector[23]=1.186270E-02;
B_vector[24]=1.012000E+00;	B_vector[25]=9.015000E-01;	B_vector[26]=1.081000E+00;	B_vector[27]=-2.239000E-01;
B_vector[28]=6.215553E-01;	B_vector[29]=-3.925198E-01;	B_vector[30]=5.820666E-01;	B_vector[31]=3.439001E-02;
B_vector[32]=1.413465E+02;	B_vector[33]=8.125362E+01;	B_vector[34]=2.686247E+02;	B_vector[35]=-9.678118E+01;
B_vector[36]=3.901087E+00;	B_vector[37]=-9.877649E-01;	B_vector[38]=2.454189E+00;	B_vector[39]=7.424383E-01;
B_vector[40]=2.997265E+02;	B_vector[41]=1.240335E+02;	B_vector[42]=5.146290E+02;	B_vector[43]=-1.742407E+02;
B_vector[44]=2.857600E+00;	B_vector[45]=2.633000E+00;	B_vector[46]=3.919000E+00;	B_vector[47]=-9.664000E-01;
B_vector[48]=9.041080E-01;	B_vector[49]=8.940613E-01;	B_vector[50]=7.836745E-01;	B_vector[51]=-2.640410E-02;
B_vector[52]=9.889809E-01;	B_vector[53]=-7.141192E-01;	B_vector[54]=1.130646E+00;	B_vector[55]=-8.393918E-02;

//Coefficients for differential cross section
C_vector[0]=3.71000E+00*E_i[0];		C_vector[1]=-2.36000E-01;	C_vector[2]=2.35000E+00;	C_vector[3]=-6.60000E-01;	C_vector[4]=0.00000E+00;
C_vector[5]=3.71000E+00*E_i[1];		C_vector[6]=-2.36000E-01;	C_vector[7]=2.35000E+00;	C_vector[8]=-6.60000E-01;	C_vector[9]=0.00000E+00;
C_vector[10]=3.71000E+00*E_i[2];	C_vector[11]=-2.36000E-01;	C_vector[12]=2.35000E+00;	C_vector[13]=-6.60000E-01;	C_vector[14]=0.00000E+00;
C_vector[15]=3.71000E+00*E_i[3];	C_vector[16]=-2.36000E-01;	C_vector[17]=2.35000E+00;	C_vector[18]=-6.60000E-01;	C_vector[19]=0.00000E+00;
C_vector[20]=3.71000E+00*E_i[4];	C_vector[21]=-2.36000E-01;	C_vector[22]=2.35000E+00;	C_vector[23]=-6.60000E-01;	C_vector[24]=0.00000E+00;
C_vector[25]=2.79400E+00*E_i[5];	C_vector[26]=4.69000E-01;	C_vector[27]=-1.29400E+01;	C_vector[28]=2.62600E+01;	C_vector[29]=-1.34300E+01;
C_vector[30]=2.79400E+00*E_i[6];	C_vector[31]=4.69000E-01;	C_vector[32]=-1.29400E+01;	C_vector[33]=2.62600E+01;	C_vector[34]=-1.34300E+01;
C_vector[35]=2.79400E+00*E_i[7];	C_vector[36]=4.69000E-01;	C_vector[37]=-1.29400E+01;	C_vector[38]=2.62600E+01;	C_vector[39]=-1.34300E+01;
C_vector[40]=2.79400E+00*E_i[8];	C_vector[41]=4.69000E-01;	C_vector[42]=-1.29400E+01;	C_vector[43]=2.62600E+01;	C_vector[44]=-1.34300E+01;
C_vector[45]=2.79400E+00*E_i[9];	C_vector[46]=4.69000E-01;	C_vector[47]=-1.29400E+01;	C_vector[48]=2.62600E+01;	C_vector[49]=-1.34300E+01;
C_vector[50]=2.41900E+00*E_i[10];	C_vector[51]=-1.32000E-01;	C_vector[52]=1.70000E+00;	C_vector[53]=0.00000E+00;	C_vector[54]=0.00000E+00;
C_vector[55]=2.41900E+00*E_i[11];	C_vector[56]=-1.32000E-01;	C_vector[57]=1.70000E+00;	C_vector[58]=0.00000E+00;	C_vector[59]=0.00000E+00;
C_vector[60]=2.41900E+00*E_i[12];	C_vector[61]=-1.32000E-01;	C_vector[62]=1.70000E+00;	C_vector[63]=0.00000E+00;	C_vector[64]=0.00000E+00;
C_vector[65]=2.41900E+00*E_i[13];	C_vector[66]=-1.32000E-01;	C_vector[67]=1.70000E+00;	C_vector[68]=0.00000E+00;	C_vector[69]=0.00000E+00;

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

 for(idx=0;idx<excitations_number*4;idx++) {B_vector[idx]*=4.831E14;}
 for(idx=0;idx<ionizations_number*5;idx++) {C_vector[idx]*=4.036E16;}
 for(idx=0;idx<ionizations_number*2;idx++) {D_vector[idx]*=3.94286E+23;}


if ((OUTPUTFILE1=fopen("Test_Ionization_Coeffs.txt", "w"))==NULL)
        {
        printf("Cannot open file! Error!\n");
        exit(2);
       	}

for(idx=0;idx<new_ion_levs;idx++)
	{

	E_i_new=E_i[current_ion_lev]*(1.0+((rand() % 2000)-1000)/10000.0);
	C_vector1=C_vector[current_ion_lev*5]*(1.0+((rand() % 2000)-1000)/10000.0);
	C_vector2=C_vector[current_ion_lev*5+1]*(1.0+((rand() % 2000)-1000)/10000.0);
	C_vector3=C_vector[current_ion_lev*5+2]*(1.0+((rand() % 2000)-1000)/10000.0);
	C_vector4=C_vector[current_ion_lev*5+3]*(1.0+((rand() % 2000)-1000)/10000.0);
	C_vector5=abs(C_vector[current_ion_lev*5+3]*(0.1+((rand() % 2000)-1000)/100000.0));
	D_vector1=D_vector[current_ion_lev*2]*(1.0+((rand() % 2000)-1000)/10000.0);
	D_vector2=D_vector[current_ion_lev*2+1]*(1.0+((rand() % 2000)-1000)/10000.0);

	fprintf(OUTPUTFILE1,"%E %E %E %E %E %E %E %E\n", E_i_new, C_vector1, C_vector2, C_vector3, C_vector4, C_vector5, D_vector1, D_vector2);
	
	current_ion_lev=(current_ion_lev+1)%ionizations_number;
	}

fclose(OUTPUTFILE1);

if ((OUTPUTFILE2=fopen("Test_Excitation_Coeffs.txt", "w"))==NULL)
        {
        printf("Cannot open file! Error!\n");
        exit(2);
       	}

for(idx=0;idx<new_ion_levs;idx++)
	{

	E_j_new=E_j[current_exc_lev]*(1.0+((rand() % 2000)-1000)/10000.0);
	B_vector1=B_vector[current_exc_lev*4]*(1.0+((rand() % 2000)-1000)/10000.0);
	B_vector2=B_vector[current_exc_lev*4+1]*(1.0+((rand() % 2000)-1000)/10000.0);
	B_vector3=B_vector[current_exc_lev*4+2]*(1.0+((rand() % 2000)-1000)/10000.0);
	B_vector4=B_vector[current_exc_lev*4+3]*(1.0+((rand() % 2000)-1000)/10000.0);

	fprintf(OUTPUTFILE2,"%E %E %E %E %E\n", E_j_new, B_vector1, B_vector2, B_vector3, B_vector4);
	
	current_exc_lev=(current_exc_lev+1)%excitations_number;
	}


fclose(OUTPUTFILE2);
}





