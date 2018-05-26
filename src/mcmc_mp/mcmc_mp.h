#ifndef __MCMC_MP_H__
#define __MCMC_MP_H__

#include "cublas_v2.h"
#include <cuda_runtime.h>

#include "resources.h"
#include "processing_util.h"
#include "alloc_util.h"
#include "lhood_mp_kernel.h"

#if defined (__cplusplus)
extern "C" {
#endif

void mp_sampler(data_str data, gsl_rng *r, mcmc_str mcin,
                  mcmc_tune_str *mct, mcmc_v_str mcdata,
                  gpu_v_str gpu, out_str *res);

void metropolis_mp(cublasHandle_t handle, gsl_rng *r, mcmc_str mcin,
                    mcmc_tune_str *mct, mcmc_v_str mcdata, mcmc_int_v mclocv, 
                    mcmc_int *mcloc, int *accepted_samples, sz_str sz,
                    gpu_v_str gpu, dev_v_str d, double *host_lhood, out_str *res);

#if defined (__cplusplus)
}
#endif

#endif // __MCMC_MP_H__