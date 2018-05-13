#ifndef __MCMC_GPU_H__
#define __MCMC_GPU_H__

#include "resources.h"
#include "alloc_util.h"
#include "processing_util.h"

void gpu_sampler(data_str data, gsl_rng *r, mcmc_str mcin,
                  mcmc_tune_str mct, mcmc_v_str mcdata,
                  out_str *out_par);

void metropolis_gpu(gsl_rng *r, mcmc_str mcin,
                    mcmc_tune_str mct, mcmc_v_str mcdata, mcmc_int_v mclocv, 
                    mcmc_int *mcloc, int *accepted_samples, sz_str sz,
                    double *dev_samples, double *dev_data, double *dev_labels,
                    double *dev_lhood, double *host_lhood);
void burn_in_metropolis_gpu(gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata, 
                            mcmc_int_v mclocv, mcmc_int *mcloc, sz_str sz,
                            double *dev_samples, double *dev_data, double *dev_labels,
                            double *dev_lhood, double *host_lhood);

void tune_ess_gpu(gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata, 
                  mcmc_int_v mclocv, mcmc_int mcloc, sz_str sz,
                  double *dev_samples, double *dev_data, double *dev_labels,
                  double *dev_lhood, double *host_lhood);
void tune_target_a_gpu_v2(gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata, 
                          mcmc_int_v mclocv, mcmc_int mcloc, sz_str sz,
                          double *dev_samples, double *dev_data, double *dev_labels,
                          double *dev_lhood, double *host_lhood);
void tune_target_a_gpu(gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata, 
                        mcmc_int_v mclocv, mcmc_int mcloc, sz_str sz,
                        double *dev_samples, double *dev_data, double *dev_labels,
                        double *dev_lhood, double *host_lhood);

void short_run_burn_in(gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, 
                        double sd, mcmc_int *mcloc, sz_str sz,
                        double *dev_samples, double *dev_data, double *dev_labels,
                        double *dev_lhood, double *host_lhood);
void short_run_metropolis(data_str data, gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, 
                          int chain_length, double sd, mcmc_int *mcloc, double *samples, 
                          int *accepted_samples, sz_str sz, double *dev_samples, 
                          double *dev_data, double *dev_labels,
                          double *dev_lhood, double *host_lhood);

double acceptance_ratio(mcmc_int_v mclocv, mcmc_int *mcloc, mcmc_str mcin, double lhood);
double log_prior(double *sample, mcmc_str mcin);

void print_gpu_info();

#endif  //__MCMC_GPU_H__