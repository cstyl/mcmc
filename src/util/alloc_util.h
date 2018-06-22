#ifndef __ALLOC_UTIL_H__
#define __ALLOC_UTIL_H__

#include "resources.h"

#if defined (__cplusplus)
extern "C" {
#endif

void malloc_data_vectors(data_str *data, mcmc_str mcin);
void malloc_data_vectors_mp(data_str *data, mcmc_str mcin);
void malloc_sample_vectors(mcmc_v_str *mcdata, mcmc_str mcin);
void malloc_mcmc_vectors(mcmc_int_v *mclocv, mcmc_str mcin);

void free_data_vectors(data_str data, mcmc_str mcin);
void free_sample_vectors(mcmc_v_str mcdata);
void free_mcmc_vectors(mcmc_int_v mclocv, mcmc_str mcin);

void malloc_autocorrelation_vectors(sec_v_str *secv, sec_str sec);
void free_autocorrelation_vectors(sec_v_str secv);

void init_rng(gsl_rng **r);
void init_rng_seed(gsl_rng **r, unsigned long int seed);
void free_rng(gsl_rng *r);

void malloc_data_vectors_sp(data_str *data, mcmc_str mcin);
void malloc_sample_vectors_sp(mcmc_v_str *mcdata, mcmc_str mcin);
void malloc_autocorrelation_vectors_sp(sec_v_str *secv, sec_str sec);
void malloc_mcmc_vectors_sp(mcmc_int_v *mclocv, mcmc_str mcin);
void free_data_vectors_sp(data_str data, mcmc_str mcin);
void free_sample_vectors_sp(mcmc_v_str mcdata);
void free_autocorrelation_vectors_sp(sec_v_str secv);
void free_mcmc_vectors_sp(mcmc_int_v mclocv, mcmc_str mcin);

#if defined (__cplusplus)
}
#endif

#endif  //__ALLOC_UTIL_H__