This code is intended to demonstrate numerical quadratures on GPUs, in particular the efficient calculation of atomic rates by integrals over the Fermi-Dirac distribution. A simple collisional-radiative model is presented here. The code is split into a CPU-only and CPU+GPU parts. Simple makefiles are provided. Please note that the paths to libraries in the CUDA makefiles should be changed by the user.

For details, contact Valentin Aslanyan.

The following files are included here:

---------------------------
| Generic GPU integration |
---------------------------

Gauss_Integration_Setup.c    -	When compiled and run, supplying an integer N as an argument, creates a file Gauss_Integration.txt containing the nodes and weights for the N-point Gauss-Legendre quadrature method. This can be pre-generated once and thereafter read by the function gauss_integration_setup_fromfile in Rate_Functions_CPU.c. Requires LAPACKE and CBLAS libraries. Functions which generate the 32-point method from a lookup table - gauss_integration_setup32 for double precision and gauss_integration_setup32_f for single precision - are automatically included in the following files, avoiding the need for these libraries.

make_Gauss_Integration_Setup -  makefile for Gauss_Integration_Setup.c, compiling with GCC requiring LAPACKE and BLAS.

Quadrature.cu		     -	Example of numerical integration on a GPU using the Gauss-Legendre method. Takes 3 (optional) arguments: Number of 1D integrals; Number of 2D integrals; Number of warps per GPU thread.

make_Quadrature		     -	makefile for Quadrature.cu, compiling with nvcc (nVidia compiler). May optionally add --use_fast_math for the fast math option.



---------------------------
|  CPU only Atomic Rates  |
---------------------------

Rate_Functions_CPU.h	-	Header file.

Rate_Functions_CPU.c	-  	Contains all the appropriate functions to carry out integrals, create rate matrices and solve the rate equations using an RK4 routine. Note that this file also contains generic functions used in this model, e.g. to find the chemical potential.

CRM_CPU.c		-  	Main file implementing a time-dependent collisional radiative model using CPU-only routines.

make_CPU		-	makefile for CRM_CPU.c, compiling with GCC requiring BLAS.

CRM_Maxwellian.c	-	Analogous to CRM.c, but solves for rates using analytical integrals over the Maxwell-Boltzmann distribution (all blocking factors are equal to 1). The integrals over the Bose-Einstein distribution are non-analytic and carried out numerically. Note that this is included for comparison only and has not been efficiently implemented on a GPU.

make_Maxwellian		-	analogous to make_CPU.



---------------------------
| CPU +  GPU Atomic Rates |
---------------------------

Rate_Functions_GPU.h	-	Header file

Rate_Functions_GPU.cu	-	Contains functions specific only to the GPU, including carrying out integrals and their use by a separate RK4 routine.

CRM.cu			-  	Main file implementing time-dependent collisional radiative model, identical to the CPU-only version, but the rate coefficients are calculated at each timestep on the GPU and memory-copied to main memory to carry out RK4 on the CPU

make_GPU		-	makefile for CRM.cu, compiling with nvcc (nVidia compiler) and requiring also the same libraries as make_CPU as well as CUDA libraries. Paths to those libraries may need to be added by the user.

---------------------------
| Benchmarking GPU code   |
---------------------------

We include several standalone files to benchmark the computation speed of various GPUs in single and double precision.

Generate_Test_Levels.c	-	Create fictitious atomic levels and cross-sectional data (for 100 000 collisional ionization, excitation and photionization transitions) to allow benchmarking an arbitrarily large atomic model. Outputs data into Test_Ionization_Coeffs.txt and Test_Excitation_Coeffs.txt.

The following files use the generated transition data to carry out the rate integrals. All files take 3 (optional) arguments: Number of excitations; Number of ionizations; Number of warps per GPU thread.

BenchmarkDouble.c	-	Uses double precision routines.

BenchmarkFloatShared.c	-	Uses single precision routines and uses shared memory to sum integrands.

BenchmarkFloatShuffle.c	-	Uses single precision routines, but uses the single precision shuffle operation to sum integrands.

