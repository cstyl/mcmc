#ifndef __RESOURCES_H__
#define __RESOURCES_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>  // to obtain various pdf 
#include <gsl/gsl_cblas.h> // to perform dot products and vector operations

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

#endif  //__RESOURCES_H__