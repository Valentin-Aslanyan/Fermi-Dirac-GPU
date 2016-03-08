This code is intended to demonstrate numerical quadratures on GPUs, in particular the efficient calculation of atomic rates by integrals over the Fermi-Dirac distribution. A simple collisional-radiative model is presented here. The code is split into a CPU-only and CPU+GPU parts. Simple makefiles are provided. Please note that the paths to libraries in the CUDA makefiles should be changed by the user.

For details, contact Valentin Aslanyan.

The following files are included here:

---------------------------
| Generic GPU integration |
---------------------------

Quadrature.cu		-	Example of numerical integration on a GPU using the Gauss-Legnedre method.

make_Quadrature		-	makefile for Quadrature.cu, compiling with nvcc (nVidia compiler)



---------------------------
|  CPU only Atomic Rates  |
---------------------------

Rate_Functions_CPU.h	-	Header file

Rate_Functions_CPU.c	-  	Contains all the appropriate functions to carry out integrals, create rate matrices and solve the rate equations using an RK4 routine. Note that this file also contains generic functions used in this model, e.g. to find the chemical potential

CRM_CPU.c		-  	Main file implementing a time-dependent collisional radiative model using CPU-only routines

make_CPU		-	makefile for CRM_CPU.c, compiling with GCC and using LAPACKE libraries + dependencies

CRM_Maxwellian.c	-	analogous to CRM.c, but solves for rates using analytical integrals over the Maxwell-Boltzmann distribution (all blocking factors are equal to 1). The integrals over the Bose-Einstein distribution are non-analytic and carried out numerically. Note that this is included for comparison only and has not been efficiently implemented on a GPU

make_Maxwellian		-	analogous to make_CPU



---------------------------
| CPU +  GPU Atomic Rates |
---------------------------

Rate_Functions_GPU.h	-	Header file

Rate_Functions_GPU.cu	-	Contains functions specific only to the GPU, including carrying out integrals and their use by a separate RK4 routine.

CRM.cu			-  	Main file implementing time-dependent collisional radiative model, identical to the CPU-only version, but the rate coefficients are calculated at each timestep on the GPU and memory-copied to main memory to carry out RK4 on the CPU

make_GPU		-	makefile for CRM.cu, compiling with nvcc (nVidia compiler) and requiring also the same libraries as make_CPU as well as CUDA libraries. Paths to those libraries may need to be changed by the user
