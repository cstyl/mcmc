#ifndef __MCMC_CPU_H__
#define __MCMC_CPU_H__

// typedef struct output_vars
// {
//   double acc_ratio;
//   double time_m;
//   double time_s;
//   double time_ms;
//   double ess_shift;
//   double ess_circular;
// } out_struct;

// typedef struct input_vars
// {
//   int d_data;   // dimensionality of datapoints
//   int Nd;       // number of data points
//   int Ns;       // number of samples generated
//   int burn_in;  // number of samples burned
// } in_struct;

double log_likelihood(double *sample, int data_dim, int datapoints);
double log_prior(double *sample, int data_dim);
double acceptance_ratio(double *sample, int data_dim, int datapoints);

void Metropolis_Hastings_cpu(in_struct in_par, out_struct *out_par,
                              double rw_sd);

double current_posterior, proposed_posterior;

#endif  //__MCMC_CPU_H__