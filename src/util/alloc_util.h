#ifndef __ALLOC_UTIL_H__
#define __ALLOC_UTIL_H__

#include "resources.h"

void malloc_data_vectors_cpu(data_str *data, mcmc_str mcin);
void malloc_data_vectors_gpu(data_str *data, mcmc_str mcin);
void malloc_sample_vectors(mcmc_v_str *mcdata, mcmc_str mcin);
void malloc_normalised_sample_vectors(mcmc_v_str *mcdata, mcmc_str mcin);
void malloc_mcmc_vectors_cpu(mcmc_int_v *mclocv, mcmc_str mcin);
void malloc_mcmc_vectors_gpu(mcmc_int_v *mclocv, mcmc_str mcin);

void free_data_vectors_cpu(data_str data);
void free_data_vectors_gpu(data_str data);
void free_sample_vectors(mcmc_v_str mcdata);
void free_norm_sample_vectors(mcmc_v_str mcdata);
void free_mcmc_vectors_cpu(mcmc_int_v mclocv);
void free_mcmc_vectors_gpu(mcmc_int_v mclocv);

void malloc_autocorrelation_vectors(sec_v_str *secv, sec_str sec);
void free_autocorrelation_vectors(sec_v_str secv, sec_str sec);

void init_rng(gsl_rng **r);
void free_rng(gsl_rng *r);


#endif  //__ALLOC_UTIL_H__