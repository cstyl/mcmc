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

#if defined (__cplusplus)
extern "C" {
#endif

enum dataread{
  RowMajor,
  ColMajor
};

enum implementation{
  CPU,
  GPU,
  MP,
  CA
};

typedef struct data_vectors
{
  double *data;
  float  *dataf;
  int8_t *labels;
  double *mvout;
  int8_t *zlabels;
  int    *zidx;
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
  int Ndmap;
  int Ns;             // number of samples generated
  int burnin;        // number of samples burned
  int tune;       // 0=no tuning, 1=tune for target acceptance, 2=tune for max ess
  int impl;     //0=CPU, 1=GPU, 2=MP, 3=CA

  int bright;
  int dark;
  int cand;
  int tdark;
} mcmc_str;

typedef struct tuning_par
{
  double rwsd;
} mcmc_tune_str;

typedef struct mcmc_internal_vectors
{
  double *proposed;
  double *current;
  float  *proposedf;
  float  *currentf;
} mcmc_int_v;

typedef struct mcmc_internal
{
  double cposterior;
  double pposterior;
  double acceptance;
  double u;
} mcmc_int;

typedef struct device_vectors
{
  double *samples;
  double *data;
  double *cuLhood;
  double *lhood;
  // specific to mp imlpementation
  float  *cuLhoodf;
  float  *samplesf;
  float  *dataf;
  int8_t *zlabels;
  int    *zidx;
  double *brightLhood;
  float  *darkLhood;
  double *resample;
  double *dLhood; // temp buffer
} dev_v_str;

typedef struct gpu_parameters
{
  int size;       // total elements to run
  int threads;    // max threads/block
  int blocks;     // number of blocks
  int maxThreads; // threads per block
  int maxBlocks;
  int kernel;     // choose kernel number
  int cpuThresh;  // current iteration
} gpu_v_str;

typedef struct sizes_struct
{
  size_t samples;
  size_t data;
  size_t cuLhood;
  size_t cuLhoodf;
  size_t lhood;

  size_t samplesf;
  size_t dataf;
  size_t zlabels;
  size_t zidx;
  size_t brightLhood;
  size_t darkLhood;
  size_t resample;
  size_t dLhood;

} sz_str;

typedef struct secondary_vectors
{
  double *shift;
  double *circ;
} sec_v_str;

typedef struct secondary_vars
{
  int fdata;
  int lagidx;
  int first;
} sec_str;

typedef struct outputs
{
  float cuTime, cuBandwidth;
  float kernelTime, kernelBandwidth;
  float gpuTime, gpuBandwidth;
  float cpuTime;

  float tuneTime;
  float burnTime;
  float mcmcTime;
  float samplerTime;
  
  int Nd, dim, kernel, blocksz;
  float sd, acceptance, ess;
  int samples, burnSamples;
  int device;

} out_str;

#if defined (__cplusplus)
}
#endif

#endif  //__RESOURCES_H__