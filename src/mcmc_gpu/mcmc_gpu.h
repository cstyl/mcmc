#ifndef __MCMC_GPU_H__
#define __MCMC_GPU_H__

#include "cublas_v2.h"
#include <cuda_runtime.h>

#include "resources.h"
#include "processing_util.h"
#include "alloc_util.h"
#include "mcmc_gpu_kernel.h"
#include "gpu_util.h"

#if defined (__cplusplus)
extern "C" {
#endif

void gpu_sampler( data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, 
                  mcmc_v_str mcdata, gpu_v_str gpu, out_str *res
                );


void metropolis_gpu( cublasHandle_t handle, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, 
                      mcmc_v_str mcdata, mcmc_int_v mclocv, mcmc_int *mcloc, int *accepted_samples, 
                      sz_str sz, gpu_v_str gpu, dev_v_str d, double *host_lhood, out_str *res
                   );

void burn_in_metropolis_gpu( cublasHandle_t handle, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, 
                              mcmc_v_str mcdata, mcmc_int_v mclocv, mcmc_int *mcloc, sz_str sz,
                              gpu_v_str gpu, dev_v_str d, double *host_lhood
                           );


double gpu_likelihood_d( cublasHandle_t handle, mcmc_str mcin, gpu_v_str gpu, double *samples, 
                          size_t sampleSz, dev_v_str d, double *host_lhood, out_str *res
                       );


void tune_ess_gpu( cublasHandle_t handle, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, 
                   mcmc_int_v mclocv, mcmc_int mcloc,  sz_str sz, gpu_v_str gpu, dev_v_str d, 
                   double *host_lhood
                 );

void tune_target_a_gpu_v2( cublasHandle_t handle, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, 
                           mcmc_int_v mclocv, mcmc_int mcloc, sz_str sz, gpu_v_str gpu, 
                           dev_v_str d, double *host_lhood
                          );

void tune_target_a_gpu( cublasHandle_t handle, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, 
                        mcmc_int_v mclocv, mcmc_int mcloc, sz_str sz, gpu_v_str gpu, dev_v_str d, 
                        double *host_lhood
                      );


void short_run_burn_in_gpu( cublasHandle_t handle, gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, 
                            double sd, mcmc_int *mcloc, sz_str sz,  gpu_v_str gpu, dev_v_str d, 
                            double *host_lhood
                          );

void short_run_metropolis_gpu( cublasHandle_t handle, gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, 
                               int chain_length, double sd, mcmc_int *mcloc, double *samples, 
                               int *accepted_samples, sz_str sz, gpu_v_str gpu, dev_v_str d, 
                               double *host_lhood
                             );


double acceptance_ratio_gpu(mcmc_int_v mclocv, mcmc_int *mcloc, mcmc_str mcin, double plhood);

double log_prior_gpu(double *sample, mcmc_str mcin);

void print_gpu_info();

#if defined (__cplusplus)
}
#endif

#endif  //__MCMC_GPU_H__