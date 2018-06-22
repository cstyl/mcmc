#ifndef __MCMC_MP_H__
#define __MCMC_MP_H__

#include <cuda_runtime.h>
// #include <curand_kernel.h>
#include <curand.h>
#include "cublas_v2.h"

#include "resources.h"
#include "processing_util.h"
#include "alloc_util.h"
#include "reduction.h"

#if defined (__cplusplus)
extern "C" {
#endif

void mp_sampler(data_str data, gsl_rng *r, mp_str *mpVar, mcmc_str mcin,
                mcmc_tune_str mct, mcmc_v_str mcdata, double lhoodBound, out_str *res);

void metropolis_mp(curandGenerator_t gen, gsl_rng *r, mp_str *mpVar, mcmc_str mcin, mcmc_tune_str mct, 
                    mcmc_v_str mcdata, mcmc_int_v mclocv, mcmc_int *mcloc, int *accepted_samples,
                    dev_v_str d, double *host_lhood, float *host_lhoodf, int *dataCounters, double lhoodBound, 
                    out_str *res);
void burn_in_metropolis_mp(curandGenerator_t gen, gsl_rng *r, mp_str *mpVar, mcmc_str mcin, mcmc_tune_str mct, 
                          mcmc_v_str mcdata, mcmc_int_v mclocv, mcmc_int *mcloc, dev_v_str d, 
                          double *host_lhood, float *host_lhoodf, int *dataCounters, double lhoodBound);
double mp_likelihood(curandGenerator_t gen, gsl_rng *r, mp_str *mpVar, mcmc_str mcin, 
                     double *samples, size_t sampleSz, float *samplesf, size_t sampleSzf, 
                     dev_v_str d, double *host_lhood, float *host_lhoodf,
                     int *dataCounters, double lhoodBound, out_str *res);

double reduction_mp_d(gpu_v_str gpu, dev_v_str d, double *host_lhood, double *ke_acc_Bytes);
float reduction_mp_f(gpu_v_str gpu, dev_v_str d, float *host_lhoodf, double *ke_acc_Bytes);


double acceptance_ratio_mp(mcmc_int_v mclocv, mcmc_int *mcloc, mcmc_str mcin, double plhood); 
double log_prior_mp(double *sample, mcmc_str mcin);


double get_lowerbound(data_str data, double *initialCond, gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct);
double likelihood_bound(gsl_rng *r, mcmc_str mcin, double *samples_d, size_t sampleSz, 
                        float *samples_f, size_t sampleSzf, dev_v_str d, 
                        double *host_lhood, double *epsilon);
double call_boundKernel(mcmc_str mcin, dev_v_str d, double *host_lhood, double *epsilon);


void get_blockSz(data_str data, gsl_rng *r, mp_str *mpVar, mcmc_str mcin, 
                 mcmc_tune_str mct, double *initialCond, double lhoodBound);
void short_run_mp(curandGenerator_t gen, gsl_rng *r, mp_str *mpVar, mcmc_str mcin, mcmc_tune_str mct, 
                  double *initialCond, mcmc_int_v mclocv, mcmc_int *mcloc, dev_v_str d,
                  double *host_lhood, float *host_lhoodf, int *dataCounters, double lhoodBound);

#if defined (__cplusplus)
}
#endif

#endif // __MCMC_MP_H__