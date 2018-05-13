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

typedef struct data_vectors
{
  double *data;
  int8_t *labels;
  double *gpudata;
  int8_t *gpulabels;
} data_str;

typedef struct mcmc_vectors
{
  double *samples;  // contains the samples in linearised space
  double *nsamples;  // contains the normalised samples in linearised space
  double *burn; // contains the burn samples in linearised space
  double *nburn; // contains the normalised burn samples in linearised space   
  double *sample_means;  
} mcmc_v_str;

typedef struct mcmc_vars
{
  int ddata;         // actual dimensionality of datapoints
  int Nd;             // number of data points
  int Ns;             // number of samples generated
  int dmap;       // make dimensionality of datapoints even
  int Ndmap;        // to accomodate datapoint number power of 2
  int burnin;        // number of samples burned
} mcmc_str;

typedef struct tuning_par
{
  double rwsd;
} mcmc_tune_str;

typedef struct mcmc_internal_vectors
{
  double *proposed;
  double *current;
} mcmc_int_v;

typedef struct mcmc_internal
{
  double cposterior;
  double pposterior;
  double acceptance;
  double u;
} mcmc_int;

typedef struct sizes_struct
{
  size_t samples_map;
  size_t samples_actual;
  size_t data;
  size_t labels;
  size_t likelihood;
} sz_str;

typedef struct secondary_vectors
{
  double *shift;
  double *circ;
} sec_v_str;

typedef struct secondary_vars
{
  int fdata;
  int fauto;
  int fout;
  int fautoout;
  int lagidx;
  int fnorm;
} sec_str;

typedef struct output_vars
{
  double acc_ratio;
  int time_m;
  int time_s;
  int time_ms;
  double ess_shift;
  double ess_circular;
  double ess;
} out_str;

#endif  //__RESOURCES_H__