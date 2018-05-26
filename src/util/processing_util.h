#ifndef __PROCESSING_UTIL_H__
#define __PROCESSING_UTIL_H__

#include "resources.h"

#if defined (__cplusplus)
extern "C" {
#endif

void getSizes_mp(sz_str *sz, mcmc_str mcin, gpu_v_str gpu);

int nextPow2(int d_data);
void getBlocksAndThreads(int kernel, int n, int maxBlocks, int maxThreads, int *blocks, int *threads);

void normalise_samples(double *raw_in, double *norm_out,
                      int data_dim, int sz);
double get_bias_mean(double *in_vec, int data_dim, int sample_sz);
double get_dim_mean(double *in_vec, int data_dim, int current_dim, int sample_sz);

double shift_autocorrelation(double *out_v, double *in_m, int data_dim, 
                              int samples, int lag_idx);
double circular_autocorrelation(double *out_v, double *in_m, int data_dim, 
                                int samples, int lag_idx);
double get_ess(double *samples, int sample_sz, int sample_dim, 
                int max_lag, double *autocorrelation_v);


void get_mean(int data_dim, int samples, 
              double *data, double *mu);
void get_variance(int data_dim, int samples,
                  double *data, double *var, double *mu);

void calculate_normalised_sample_means(mcmc_v_str mcdata, mcmc_str mcin);

#if defined (__cplusplus)
}
#endif

#endif  //__PROCESSING_UTIL_H__