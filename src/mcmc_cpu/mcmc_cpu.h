#ifndef __MCMC_CPU_H__
#define __MCMC_CPU_H__

#include "resources.h"

double log_likelihood(double *sample, double *x, double *y,
                      int data_dim, int datapoints);
double log_prior(double *sample, int data_dim);
double acceptance_ratio(double *sample, double *x, double *y, 
                        int data_dim, int datapoints);

void Metropolis_Hastings_cpu(double *x, double *y, gsl_rng *r,
                            in_struct in_par, out_struct *out_par, 
                            double rw_sd, double *samples_m, double *burn_m);

double current_posterior, proposed_posterior;

#endif  //__MCMC_CPU_H__