/*
 * Implementation of MCMC Metropolis-Hastings Algorithm
 * using CPU for processing
 */
#ifndef __MCMC_CPU_C__
#define __MCMC_CPU_C__

#include "mcmc_cpu.h"

const int PRIOR_SD = 10;

void cpu_sampler(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, 
                  mcmc_v_str mcdata, out_str *res)
{
  mcmc_int_v mclocv;
  mcmc_int mcloc;
  malloc_mcmc_vectors(&mclocv, mcin);
  
  int accepted_samples;
  clock_t startBurn, stopBurn;
  clock_t startMcmc, stopMcmc;

  startBurn = clock();
  burn_in_metropolis_cpu(data, r, mcin, mct, mcdata, mclocv, &mcloc);
  stopBurn = clock() - startBurn;

  accepted_samples = 0;
   
  startMcmc = clock();
  metropolis_cpu(data, r, mcin, mct, mcdata, mclocv, &mcloc, &accepted_samples, res);
  stopMcmc = clock() - startMcmc;

  res->burnTime = stopBurn * 1000 / CLOCKS_PER_SEC;   // burn in time in ms
  res->mcmcTime = stopMcmc * 1000 / CLOCKS_PER_SEC;   // mcmc time in ms
  res->acceptance = (double)accepted_samples / mcin.Ns;

  free_mcmc_vectors(mclocv, mcin);
}

void metropolis_cpu(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata,
                    mcmc_int_v mclocv, mcmc_int *mcloc, int *accepted_samples, out_str *res)
{
  int i, dim_idx;
  double plhood;
  res->cpuTime = 0;

  fprintf(stdout, "Starting metropolis algorithm. Selected rwsd = %f\n", mct.rwsd); 

  for(i=0; i<mcin.Ns; i++)
  {
    // propose next sample
    for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
      mclocv.proposed[dim_idx] = mclocv.current[dim_idx] + gsl_ran_gaussian_ziggurat(r, mct.rwsd); // random walk using Marsaglia-Tsang ziggurat algorithm
    
    plhood = log_likelihood(mclocv.proposed, data, mcin, res);

    mcloc->acceptance = acceptance_ratio(mclocv, mcloc, mcin, plhood);
    mcloc->u = gsl_rng_uniform(r);

    if(log(mcloc->u) <= mcloc->acceptance)
    {
      // accept proposed sample
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
      {
        mcdata.samples[i*mcin.ddata + dim_idx] = mclocv.proposed[dim_idx];
        mclocv.current[dim_idx] = mclocv.proposed[dim_idx];
      }
      mcloc->cposterior = mcloc->pposterior;
      *accepted_samples += 1;
    }else{
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
        mcdata.samples[i*mcin.ddata + dim_idx] = mclocv.current[dim_idx];
    }    
  } 
  fprintf(stdout, "Metropolis algorithm finished. Accepted Samples = %d\n\n", *accepted_samples);
}

void burn_in_metropolis_cpu(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata,
                            mcmc_int_v mclocv, mcmc_int *mcloc)
{
  int i, dim_idx;
  double clhood, plhood;
  out_str res;

  fprintf(stdout, "Starting burn in process. Selected rwsd = %f\n", mct.rwsd);

  // initialize burn in sequence
  for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
    mclocv.current[dim_idx] = mcdata.burn[dim_idx];

  clhood = log_likelihood(mclocv.current, data, mcin, &res);

  // calculate the current posterior
  mcloc->cposterior = log_prior(mclocv.current, mcin) + clhood;

  // start burn in
  for(i=1; i<mcin.burnin; i++)
  {
    // propose next sample
    for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
      mclocv.proposed[dim_idx] = mclocv.current[dim_idx] + gsl_ran_gaussian_ziggurat(r, mct.rwsd); // random walk using Marsaglia-Tsang ziggurat algorithm
    
    plhood = log_likelihood(mclocv.proposed, data, mcin, &res);   

    mcloc->acceptance = acceptance_ratio(mclocv, mcloc, mcin, plhood);
    mcloc->u = gsl_rng_uniform(r);

    if(log(mcloc->u) <= mcloc->acceptance)
    {
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
      {
        mcdata.burn[i*mcin.ddata + dim_idx] = mclocv.proposed[dim_idx];
        mclocv.current[dim_idx] = mclocv.proposed[dim_idx];
      }
      mcloc->cposterior = mcloc->pposterior;
    }else{
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
        mcdata.burn[i*mcin.ddata + dim_idx] = mclocv.current[dim_idx];
    }
  }
  fprintf(stdout, "Burn in process finished.\n\n");
}

// tune rwsd for a target acceptance ratio
void tune_ess_cpu(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, double *initCond, int length)
{
  mcmc_int_v mclocv;
  mcmc_int mcloc;
  malloc_mcmc_vectors(&mclocv, mcin);

  int chain_length = length;
  int runs = 40;
  double target_a[] = {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50};
  double error_tolerance = 0.01;
  double min_error = 9999999;
  double max_ess = -9999999;
  double lagidx = 500;

  double sd = mct->rwsd;
  double ess_sd = sd;

  int accepted_samples, run, a_idx;
  double acc_ratio_c, acc_error_c, best_acc_ratio;
  double circ_sum, best_sd, ess_c;

  double *samples = NULL;
  samples = (double*) malloc(mcin.ddata * chain_length * sizeof(double));
  if(samples == NULL)
    fprintf(stderr, "ERROR: Samples vector did not allocated.\n");
  double *autocorr_lagk = NULL;
  autocorr_lagk = (double*) malloc(lagidx * sizeof(double));
  if(autocorr_lagk == NULL)
    fprintf(stderr, "ERROR: Autocorrelation vector did not allocated.\n");

  fprintf(stdout, "\nStarting tuning process. Rwsd = %5.3f\n", sd);
  
  for(a_idx=0; a_idx<9; a_idx++){
    fprintf(stdout, "\tStarting tuning for target ratio = %4.3f. Current rwsd = %5.3f\n", target_a[a_idx], sd);    
    min_error = 9999999;
    for(run=0; run<runs; run++)
    {
      fprintf(stdout, "\t\tStarting Run %2d. Current rwsd = %5.3f\n", run, sd);
      accepted_samples = 0;
      short_run_burn_in(data, r, mclocv, mcin, sd, &mcloc, initCond);
      short_run_metropolis(data, r, mclocv, mcin, chain_length, sd, 
                            &mcloc, samples, &accepted_samples);
      
      acc_ratio_c = accepted_samples/(double)chain_length;
      acc_error_c = fabs(acc_ratio_c - target_a[a_idx]);

      if(acc_error_c < min_error) // accept the current sd
      {
        best_sd = sd;
        min_error = acc_error_c;
        best_acc_ratio = acc_ratio_c;
        fprintf(stdout, "\t\t\tAccepted: rwsd = %5.3f, acceptance = %4.3f, error = %4.3f\n", 
                        best_sd, best_acc_ratio, min_error);
      }else{
        fprintf(stdout, "\t\t\trwsd = %5.3f, acceptance = %4.3f, error = %4.3f\n", 
                          sd, acc_ratio_c, acc_error_c);
      }
      
      if(min_error < error_tolerance) 
        break;
      
      sd *= acc_ratio_c/target_a[a_idx];
    }
    
    circ_sum = circular_autocorrelation(autocorr_lagk, samples, mcin.ddata,
                                        chain_length, lagidx);
    ess_c = chain_length / (1 + 2*circ_sum);
    
    if(ess_c > max_ess)
    {
      max_ess = ess_c;
      ess_sd = sd;
      fprintf(stdout, "\tAccepted: ess = %8.3f, rwsd = %5.3f\n", max_ess, ess_sd);
    }else{
      fprintf(stdout, "\tess= %8.3f, rwsd = %5.3f\n", ess_c, sd);
    }
  }
  mct->rwsd = ess_sd;
  fprintf(stdout, "Tuning finished. Selected rwsd = %5.3f\n\n", mct->rwsd);
  
  free(samples);
  free(autocorr_lagk);
  free_mcmc_vectors(mclocv, mcin);
}


// tune rwsd for a target acceptance ratio
void tune_target_a_cpu_v2(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, double *initCond, double ratio, int max_reps)
{
  mcmc_int_v mclocv;
  mcmc_int mcloc;
  malloc_mcmc_vectors(&mclocv, mcin);

  int chain_length = 5000;
  int runs = max_reps;
  double target_a = ratio;
  double error_tolerance = 0.01;
  double min_error = 9999999;

  double sd = mct->rwsd;
  double best_sd = sd;

  int accepted_samples, run;
  double acc_ratio_c, acc_error_c, best_acc_ratio;

  double *samples = NULL;
  samples = (double*) malloc(mcin.ddata * chain_length * sizeof(double));
  if(samples == NULL)
    fprintf(stderr, "ERROR: Samples vector did not allocated.\n");

  fprintf(stdout, "\nStarting tuning process. Rwsd = %5.3f\n", sd);
  
  for(run=0; run<runs; run++)
  {
    fprintf(stdout, "\tStarting Run %2d. Current rwsd = %5.3f\n", run, sd);
    accepted_samples = 0;
    short_run_burn_in(data, r, mclocv, mcin, sd, &mcloc, initCond);
    short_run_metropolis(data, r, mclocv, mcin, chain_length, sd, 
                          &mcloc, samples, &accepted_samples);
    
    acc_ratio_c = accepted_samples/(double)chain_length;
    acc_error_c = fabs(acc_ratio_c - target_a);

    if(acc_error_c < min_error) // accept the current sd
    {
      best_sd = sd;
      min_error = acc_error_c;
      best_acc_ratio = acc_ratio_c;
      fprintf(stdout, "\t\tAccepted: rwsd = %5.3f, acceptance = %4.3f, error = %4.3f\n", 
                      best_sd, best_acc_ratio, min_error);
    }else{
      fprintf(stdout, "\t\trwsd = %5.3f, acceptance = %4.3f, error = %4.3f\n", 
                        sd, acc_ratio_c, acc_error_c);
    }
    
    if(min_error < error_tolerance) 
      break;
    
    sd *= acc_ratio_c/target_a;
  }

  mct->rwsd = best_sd;
  fprintf(stdout, "Tuning finished. Selected rwsd = %5.3f\n\n", mct->rwsd);
  
  free(samples);
  free_mcmc_vectors(mclocv, mcin);
}

// tune rwsd for a target acceptance ratio
void tune_target_a_cpu(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, double *initCond)
{
  mcmc_int_v mclocv;
  mcmc_int mcloc;
  malloc_mcmc_vectors(&mclocv, mcin);

  int chain_length = 5000;
  int run = 0;
  double target_a = 0.25;
  double error_tolerance = 0.01;
  double mult_factor = 0.1;

  double sd = mct->rwsd;

  int accepted_samples;
  double acc_ratio_c;

  double *samples = NULL;
  samples = (double*) malloc(mcin.ddata * chain_length * sizeof(double));
  if(samples == NULL)
    fprintf(stderr, "Samples vector did not allocated.\n");

  fprintf(stdout, "\nStarting tuning process. Rwsd = %5.3f\n", sd);
  
  while(1)
  {
    fprintf(stdout, "\tStarting Run %2d. Current rwsd = %5.3f, Acceptance = ", run, sd);
    accepted_samples = 0;
    short_run_burn_in(data, r, mclocv, mcin, sd, &mcloc, initCond);
    short_run_metropolis(data, r, mclocv, mcin, chain_length, sd, 
                          &mcloc, samples, &accepted_samples);
    
    acc_ratio_c = accepted_samples/(double)chain_length;
    fprintf(stdout, "%4.3f\n", acc_ratio_c);

    if(acc_ratio_c > target_a + error_tolerance){
      sd *= (1+mult_factor);
    }else if(acc_ratio_c < target_a - error_tolerance){
      sd *= (1-mult_factor);
    }else{
      break;
    }
    run++;    
  }
  
  mct->rwsd = sd;
  fprintf(stdout, "Tuning finished. Selected rwsd = %5.3f\n\n", mct->rwsd);
  
  free(samples);
  free_mcmc_vectors(mclocv, mcin);
}

void short_run_burn_in(data_str data, gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, double sd, mcmc_int *mcloc, double *initCond)
{
  int i, dim_idx;
  double clhood, plhood;
  out_str res;

  // initialize burn in sequence
  for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
    mclocv.current[dim_idx] = initCond[dim_idx];
  
  clhood = log_likelihood(mclocv.current, data, mcin, &res);
  mcloc->cposterior = log_prior(mclocv.current, mcin) + clhood;

  // start burn-in
  for(i=1; i<mcin.burnin; i++)
  {
    for(dim_idx = 0; dim_idx < mcin.ddata; dim_idx++){
      mclocv.proposed[dim_idx] = mclocv.current[dim_idx] 
                                  + gsl_ran_gaussian_ziggurat(r, sd); // random walk using Marsaglia-Tsang ziggurat algorithm  
    }

    plhood = log_likelihood(mclocv.proposed, data, mcin, &res);

    mcloc->acceptance = acceptance_ratio(mclocv, mcloc, mcin, plhood);   // Calculate acceptance ratio in the log domain
    mcloc->u = gsl_rng_uniform(r);

    if(log(mcloc->u) <= mcloc->acceptance)    // decide if you accept the proposed theta or not
    {
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++){
        mclocv.current[dim_idx] = mclocv.proposed[dim_idx];
      }
      mcloc->cposterior = mcloc->pposterior; // make proposed posterior the current 
    }
  }
}

void short_run_metropolis(data_str data, gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, int chain_length, 
                          double sd, mcmc_int *mcloc, double *samples, int *accepted_samples)
{
  int i, dim_idx;
  double plhood;
  out_str res;

  // start metropolis
  for(i=0; i < chain_length; i++){    
    for(dim_idx = 0; dim_idx < mcin.ddata; dim_idx++){
      mclocv.proposed[dim_idx] = mclocv.current[dim_idx] 
                                  + gsl_ran_gaussian_ziggurat(r, sd); // random walk using Marsaglia-Tsang ziggurat algorithm    
    }

    plhood = log_likelihood(mclocv.proposed, data, mcin, &res);

    mcloc->acceptance = acceptance_ratio(mclocv, mcloc, mcin, plhood);  // Calculate acceptance ratio in the log domain
    mcloc->u = gsl_rng_uniform(r);

    if(log(mcloc->u) <= mcloc->acceptance)    // decide if you accept the proposed theta or not
    {
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++){
        mclocv.current[dim_idx] = mclocv.proposed[dim_idx];
        samples[i*mcin.ddata + dim_idx] = mclocv.proposed[dim_idx];
      }
      mcloc->cposterior = mcloc->pposterior; // make proposed posterior the current 
      *accepted_samples += 1; 
    }else{
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++){
        samples[i*mcin.ddata + dim_idx] = mclocv.current[dim_idx];
      }      
    }     
  }
}

double acceptance_ratio(mcmc_int_v mclocv, mcmc_int *mcloc, mcmc_str mcin, double lhood) 
{
  double log_ratio;

  mcloc->pposterior = log_prior(mclocv.proposed, mcin) + lhood;
  log_ratio = mcloc->pposterior - mcloc->cposterior;

  return log_ratio;
}

double log_prior(double *sample, mcmc_str mcin)
{ 
  double log_prob = 0;
  int dim_idx;

  for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++){  //assuming iid priors
    log_prob += log(gsl_ran_gaussian_pdf(sample[dim_idx], PRIOR_SD));
  }

  return log_prob;
}

double log_likelihood(double *sample, data_str data, mcmc_str mcin, out_str *res)
{
  double log_lik = 0;
  int idx;
  double alpha, beta;
  int m, n, lda, incx, incy;
  clock_t startLikelihood, stopLikelihood;

  n = mcin.ddata;
  m = mcin.Nd;
  lda = mcin.Nd;
  incx = 1;
  incy = 1;
  alpha = 1;
  beta = 0;

  startLikelihood = clock();
  cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, data.data, lda, 
              sample, incx, beta, data.mvout, incy);

  for(idx=0; idx<mcin.Nd; idx++){
    log_lik -= log(1+exp(data.mvout[idx]));
  }
  stopLikelihood = clock() - startLikelihood;
  res->cpuTime += (stopLikelihood * 1000 / CLOCKS_PER_SEC) / mcin.Ns;
  return log_lik;
}

// double log_likelihood(double *sample, data_str data, mcmc_str mcin, out_str *res)
// {
//   double log_lik = 0;
//   int idx;
  
//   for(idx=0; idx<mcin.Nd; idx++){
//     log_lik -= log(1+exp(cblas_ddot(mcin.ddata,&data.data[idx * mcin.ddata],1,sample,1)));
//   }
//   return log_lik;
// }

#endif // __MCMC_CPU_C__