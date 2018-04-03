/*
 * Implementation of MCMC Metropolis-Hastings Algorithm
 * using CPU for processing
 */
#include "cpu_host.h"
#include "mcmc_cpu.h"

const int PRIOR_SD = 10;

void Metropolis_Hastings_cpu(in_struct in_par, out_struct *out_par, 
                              double rw_sd)
{
  double *proposed_sample = (double *) malloc(in_par.d_data * sizeof(double));
  double *current_sample = (double *) malloc(in_par.d_data * sizeof(double)); 
  double *temp_sample = (double *) malloc(in_par.d_data * sizeof(double)); 

  double acceptance, u;
  int accepted_samples = 0;
  int dim_idx,i;

  clock_t start, stop;
  start  = clock();

  // initialisation
  memcpy(current_sample, sample_matrix, in_par.d_data*sizeof(double));

  current_posterior = log_prior(current_sample, in_par.d_data) 
                      + log_likelihood(current_sample, in_par.d_data, in_par.Nd);

  //perfrom metropolis-hastings algorithm
  for(i=1; i<in_par.Ns+in_par.burn_in; i++){    
    for(dim_idx = 0; dim_idx < in_par.d_data; dim_idx++){
      proposed_sample[dim_idx] = current_sample[dim_idx] 
                                  + gsl_ran_gaussian_ziggurat(r, rw_sd); // random walk using Marsaglia-Tsang ziggurat algorithm
    }

    acceptance = acceptance_ratio(proposed_sample, in_par.d_data, in_par.Nd);   // Calculate acceptance ratio in the log domain
    u = gsl_rng_uniform(r);

    if(u <= acceptance)    // decide if you accept the proposed theta or not
    {
      memcpy(temp_sample, proposed_sample, in_par.d_data*sizeof(double));
      memcpy(current_sample, proposed_sample, in_par.d_data*sizeof(double));
      current_posterior = proposed_posterior; // make proposed posterior the current
      accepted_samples += 1;   
    }else{
      memcpy(temp_sample, current_sample, in_par.d_data*sizeof(double));
    }

    if(i < in_par.burn_in){
      for(dim_idx = 0; dim_idx < in_par.d_data; dim_idx++)
        burn_matrix[(i * in_par.d_data) + dim_idx] = temp_sample[dim_idx];
    }else{
      for(dim_idx = 0; dim_idx < in_par.d_data; dim_idx++)
        sample_matrix[(i - in_par.burn_in) * in_par.d_data + dim_idx] = temp_sample[dim_idx];
    }        
  }

  stop = clock() - start;
  out_par->time_m = stop / (CLOCKS_PER_SEC * 60);
  out_par->time_s = (stop / CLOCKS_PER_SEC) - (out_par->time_m * 60);
  out_par->time_ms = (stop * 1000 / CLOCKS_PER_SEC) - (out_par->time_s * 1000) 
                      - (out_par->time_m * 1000 * 60);

  
  out_par -> acc_ratio = (double)accepted_samples / in_par.Ns;

  free(proposed_sample);
  free(current_sample);
  free(temp_sample);
}

/************************************ PRIVATE FUNCTIONS **************************************/
double acceptance_ratio(double *sample, int data_dim, int datapoints)
{
  double log_ratio;
  proposed_posterior = log_prior(sample, data_dim) + log_likelihood(sample, data_dim, datapoints);
  log_ratio = proposed_posterior - current_posterior;
  return exp(log_ratio);
}

double log_prior(double *sample, int data_dim)
{ 
  double log_prob = 0;
  int dim_idx;

  for(dim_idx=0; dim_idx<data_dim; dim_idx++){  //assuming iid priors
    log_prob += log(gsl_ran_gaussian_pdf(sample[dim_idx], PRIOR_SD));
  }
  return log_prob;
}

/* Calculate log-likelihood for each data-point and accumulate the contributions */
double log_likelihood(double *sample, int data_dim, int datapoints)
{
  double log_lik = 0;
  int idx;
  
  for(idx=0; idx<datapoints; idx++){
    log_lik -= log(1+exp(-y[idx] * cblas_ddot(data_dim,&x[idx*data_dim],1,sample,1)));
  }
  return log_lik;
}






