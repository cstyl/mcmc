#ifndef __MCMC_GPU_H__
#define __MCMC_GPU_H__

#include "cublas_v2.h"
#include <cuda_runtime.h>

#include "resources.h"
#include "processing_util.h"
#include "alloc_util.h"
#include "reduction.h"

#if defined (__cplusplus)
extern "C" {
#endif

void gpu_sampler(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, 
                  mcmc_v_str mcdata, gpu_v_str gpu, out_str *res);

void metropolis_gpu(cublasHandle_t handle, gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata, 
                    mcmc_int_v mclocv, mcmc_int *mcloc, int *accepted_samples, sz_str sz, 
                    gpu_v_str gpu, dev_v_str d, double *host_lhood, out_str *res);
void burn_in_metropolis_gpu(cublasHandle_t handle, gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata, 
                              mcmc_int_v mclocv, mcmc_int *mcloc, sz_str sz, gpu_v_str gpu, 
                              dev_v_str d, double *host_lhood);


double reduction_d(gpu_v_str gpu, dev_v_str d, double *host_lhood, double *ke_acc_Bytes);
double gpu_likelihood_d(cublasHandle_t handle, mcmc_str mcin, gpu_v_str gpu, double *samples, size_t sampleSz, 
                        dev_v_str d, double *host_lhood, out_str *res);


void tune_ess_gpu(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, gpu_v_str gpu, double *initCond, int length);
void tune_target_a_gpu_v2(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, 
                          gpu_v_str gpu, double *initCond, double ratio, int max_reps);
void tune_target_a_gpu(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, gpu_v_str gpu, double *initCond, int length);


void short_run_burn_in_gpu(cublasHandle_t handle, gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, double sd, mcmc_int *mcloc, 
                            sz_str sz, gpu_v_str gpu, dev_v_str d, double *host_lhood, double *initCond);
void short_run_metropolis_gpu(cublasHandle_t handle, gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, int chain_length, double sd, 
                              mcmc_int *mcloc, double *samples, int *accepted_samples, sz_str sz, 
                              gpu_v_str gpu, dev_v_str d, double *host_lhood);


double acceptance_ratio_gpu(mcmc_int_v mclocv, mcmc_int *mcloc, mcmc_str mcin, double plhood);

double log_prior_gpu(double *sample, mcmc_str mcin);

void print_gpu_info();

#if defined (__cplusplus)
}
#endif

#endif  //__MCMC_GPU_H__