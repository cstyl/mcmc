#ifndef __PROCESSING_UTIL_H__
#define __PROCESSING_UTIL_H__

#include "resources.h"

void normalise_samples(double *raw_in, double *norm_out,
                      int data_dim, int sz, double bias_mu);
double get_bias_mean(double *in_vec, int data_dim, int sample_sz);
double get_dim_mean(double *in_vec, int data_dim, int current_dim, int sample_sz);

double shift_autocorrelation(double *out_v, double *in_m, int data_dim, 
                              int samples, int lag_idx);
double circular_autocorrelation(double *out_v, double *in_m, int data_dim, 
                                int samples, int lag_idx);

void get_mean(int data_dim, int samples, 
              double *data, double *mu);
void get_variance(int data_dim, int samples,
                  double *data, double *var, double *mu);
void map_gpu_data(data_str data, mcmc_str mcin);
int map_dimensions(int d_data);
#endif  //__PROCESSING_UTIL_H__