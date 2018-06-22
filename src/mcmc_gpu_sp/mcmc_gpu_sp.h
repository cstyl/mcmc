#ifndef __MCMC_GPU_SP_H__
#define __MCMC_GPU_SP_H__

#include "cublas_v2.h"
#include <cuda_runtime.h>

#include "resources.h"
#include "processing_util.h"
#include "alloc_util.h"
#include "reduction.h"

#if defined (__cplusplus)
extern "C" {
#endif

void sp_sampler(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, 
                  mcmc_v_str mcdata, gpu_v_str gpu, out_str *res);


void metropolis_sp(cublasHandle_t handle, gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata, 
                    mcmc_int_v mclocv, mcmc_int *mcloc, int *accepted_samples, sz_str sz, 
                    gpu_v_str gpu, dev_v_str d, float *host_lhoodff, out_str *res);

void burn_in_metropolis_sp(cublasHandle_t handle, gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata, 
                              mcmc_int_v mclocv, mcmc_int *mcloc, sz_str sz, gpu_v_str gpu, 
                              dev_v_str d, float *host_lhoodf);


float reduction_f(gpu_v_str gpu, dev_v_str d, float *host_lhoodf, float *ke_acc_Bytes);

float gpu_likelihood_f(cublasHandle_t handle, mcmc_str mcin, gpu_v_str gpu, float *samplesf, float sampleSz, 
                        dev_v_str d, float *host_lhoodf, out_str *res);


void tune_ess_sp(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, gpu_v_str gpu, float *initCond, int length);

void tune_target_a_sp_v2(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, 
                          gpu_v_str gpu, float *initCond, float ratio, int max_reps);


void short_run_burn_in_sp(cublasHandle_t handle, gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, float sd, mcmc_int *mcloc, 
                            sz_str sz, gpu_v_str gpu, dev_v_str d, float *host_lhoodf, float *initCond);

void short_run_metropolis_sp(cublasHandle_t handle, gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, int chain_length, float sd, 
                              mcmc_int *mcloc, float *samples, int *accepted_samples, sz_str sz, 
                              gpu_v_str gpu, dev_v_str d, float *host_lhoodf);


float acceptance_ratio_sp(mcmc_int_v mclocv, mcmc_int *mcloc, mcmc_str mcin, float plhood);

float log_prior_sp(float *sample, mcmc_str mcin);

#if defined (__cplusplus)
}
#endif

#endif  //__MCMC_GPU_SP_H__