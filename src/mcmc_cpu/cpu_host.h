#ifndef __CPU_HOST_H__
#define __CPU_HOST_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

#include <gsl/gsl_randist.h>  // to obtain various pdf 
#include <gsl/gsl_cblas.h> // to perform dot products and vector operations
#include <gsl/gsl_rng.h>

typedef enum 
{
  NoData,
  Synthetic,
  Mnist
} data_case_t;

typedef enum 
{
  NoAut,       
  Shift,
  Circ,
  Both
} autocorr_case_t;

typedef enum 
{
  NoOut,       
  Raw,
  Normalized,
  AllOut
} out_case_t;

typedef enum 
{
  NoAutOut,       
  OnlyShift,
  OnlyCirc,
  AllAut
} autoc_out_case_t;

typedef struct output_vars
{
  double acc_ratio;
  int time_m;
  int time_s;
  int time_ms;
  double ess_shift;
  double ess_circular;
} out_struct;

typedef struct input_vars
{
  int d_data;   // dimensionality of datapoints
  int Nd;       // number of data points
  int Ns;       // number of samples generated
  int burn_in;  // number of samples burned
} in_struct;

/************************************ GLOBAL VARIABLES ***************************************/
double *x;    // data points
double *y;    // labels
double *sample_matrix, *norm_sample_matrix;  // contains the samples in linearised space
double *burn_matrix, *norm_burn_matrix;      // contains the burn samples in linearised space
double *autocorrelation_shift, 
        *autocorrelation_circ;

gsl_rng * r;    // global random number generator

#endif  //__CPU_HOST_H__