#ifndef __ALLOC_UTIL_H__
#define __ALLOC_UTIL_H__

#include "resources.h"

void malloc_data_vectors_cpu(double **data, double **labels, 
                        int data_dim, int datapoints);
// void data_vectors_gpu(int data_dim, int gpu_data_dim, int datapoints);
void malloc_sample_vectors(double **parameters, double **burn_parameters, 
                    int data_dim, int samples, int burn_samples);
void malloc_normalised_sample_vectors(double **norm_par, double **norm_burn_par, 
                              int data_dim, int samples, int burn_samples);

void free_data_vectors_cpu(double *data, double *labels);
// void free_data_vectors_gpu();
void free_sample_vectors(double *parameters, double *burn_parameters);
void free_norm_sample_vectors(double *norm_par, double *norm_burn_par);

void malloc_autocorrelation_vectors(double **auto_shift, double **auto_circ,
                              int auto_case, int lag);
void free_autocorrelation_vectors(double *auto_shift, double *auto_circ,
                                  int auto_case);

void init_rng(gsl_rng **r);
void free_rng(gsl_rng *r);


#endif  //__ALLOC_UTIL_H__