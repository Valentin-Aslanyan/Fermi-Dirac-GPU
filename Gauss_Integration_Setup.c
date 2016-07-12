#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <lapacke_mangling.h>
#include <lapacke_utils.h>

//Set up the nodes (x) and weights (w) of Gaussian quadrature, using the Golub Welsch algorithm
//coeffs and vectors are allocated arrays which are required only for solving the tridiagonal matrix equation
void gauss_integration_setup(int datapoints, double *weights, double *x)
{  int idx; double *coeffs, *vectors;

  coeffs=(double*)malloc((datapoints-1)*sizeof(double));	
  vectors=(double*)malloc(datapoints*datapoints*sizeof(double));

   x[0]=0.0;
   for (idx=1;idx<datapoints;idx++)
	{
	x[idx]=0.0;
	coeffs[idx-1]=0.5/sqrt(1.0-1.0/(4.0*idx*idx));
	}	

   //dstev finds the eigenvalues and vectors of a symmetric matrix
   LAPACKE_dstev(LAPACK_ROW_MAJOR,'v', datapoints, x, coeffs, vectors, datapoints);

   for (idx=0;idx<datapoints;idx++)
	{
	x[idx]=0.5*(x[idx]+1.0);	//This leads to nodes in the range (0,1)
	weights[idx]=vectors[idx]*vectors[idx];
	}

  free(coeffs);
  free(vectors);
}

int main(int argc, char *argv[])
{

int h_datapoints=32; double *h_w, *h_x;
if (argc>1){h_datapoints=atoi(argv[1]);}
h_x=(double*)malloc(h_datapoints*sizeof(double));		
h_w=(double*)malloc(h_datapoints*sizeof(double));
FILE *INFILE, *OUTFILE;

gauss_integration_setup(h_datapoints, h_w, h_x);

if ((OUTFILE=fopen("Gauss_Integration.txt", "w"))==NULL)
        {
        printf("Cannot open file! Error!\n");
        exit(2);
       	}
fprintf(OUTFILE,"%i\n",h_datapoints);
for (int idx=0;idx<h_datapoints;idx++)
	{
	fprintf(OUTFILE,"%.16E %.16E\n",h_x[idx],h_w[idx]);
	}
fclose(OUTFILE);

exit(0);
}

