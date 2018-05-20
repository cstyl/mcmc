#ifndef __MCMC_CPU_H__
#define __MCMC_CPU_H__

#include "resources.h"
#include "alloc_util.h"
#include "processing_util.h"

#if defined (__cplusplus)
extern "C" {
#endif

void cpu_sampler(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, 
                  mcmc_v_str mcdata, out_str *res);


void metropolis_cpu(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, mcmc_v_str mcdata,
                    mcmc_int_v mclocv, mcmc_int *mcloc, int *accepted_samples, out_str *res);
void burn_in_metropolis_cpu(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, mcmc_v_str mcdata,
                            mcmc_int_v mclocv, mcmc_int *mcloc);


void tune_ess_cpu(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct);
void tune_target_a_cpu_v2(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct);
void tune_target_a_cpu(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct);


void short_run_burn_in(data_str data, gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, double sd, mcmc_int *mcloc);
void short_run_metropolis(data_str data, gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, int chain_length, 
                            double sd, mcmc_int *mcloc, double *samples, int *accepted_samples);


double acceptance_ratio(mcmc_int_v mclocv, mcmc_int *mcloc, mcmc_str mcin, double lhood);
double log_prior(double *sample, mcmc_str mcin);
double log_likelihood(double *sample, data_str data, mcmc_str mcin, out_str *res);

#if defined (__cplusplus)
}
#endif

#endif  //__MCMC_CPU_H__