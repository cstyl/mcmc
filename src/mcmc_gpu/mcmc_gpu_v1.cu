/*
 * -Implementation of MCMC Metropolis-Hastings Algorithm
 *  using GPU for processing
 * - Likelihood Kernel:
 *    performs dot product and exp on all data points 
 *    each block represents a data point 
 *      -> max dim=1024
 *      -> blocksPerGrid = Nd
 *    returns array of L_n (size of Nd)
 */
extern "C" {
#include "mcmc_gpu.h"
}

const int PRIOR_SD = 10;

__global__ void Likelihood_v1_sequential_addressing(double *samples, double *data, int8_t *labels,
                                                      double *L_n)
{
  __shared__ double shared_mem[];
  int data_idx = blockIdx.x * blockDim.x + threadIdx.x; // linearised datapoint space
  int label_idx = blockIdx.x;   // one datapoint exists per block
  int tidx = threadIdx.x;

  // calculates and loads a single dot product to shared memory
  shared_mem[tidx] = -labels[label_idx] * data[data_idx] * samples[tidx];
  __syncthreads();

  // perform reduction using shared memory
  int i = blockDim.x/2;
  while (i != 0) {
    if (tidx < i)
      shared_mem[tidx] += shared_mem[tidx + i];
    __syncthreads();
    i /= 2; 
  }
  // write result for this block back to global mem
  if (tidx == 0) 
    L_n[blockIdx.x] = -log(1 + exp(shared_mem[0])); 
}

void gpu_sampler(data_str data, gsl_rng *r, mcmc_str mcin,
                  mcmc_tune_str mct, mcmc_v_str mcdata,
                  out_str *out_par)
{
  print_gpu_info();
  cudaSetDevice(0);

  mcmc_int_v mclocv;
  mcmc_int mcloc;
  malloc_mcmc_vectors_gpu(&mclocv, mcin);
  
  // set up the gpu vectors
  sz_str sz;
  sz.samples_map = mcin.dmap * sizeof(double);
  sz.samples_actual = mcin.ddata * sizeof(double);
  sz.data = mcin.dmap * mcin.Ndmap * sizeof(double);
  sz.labels = mcin.Ndmap * sizeof(int8_t);
  sz.likelihood = mcin.Ndmap * sizeof(double);

  double *host_lhood = (double *) malloc(sz.likelihood * sizeof(double));
  double *dev_samples, *dev_data, *dev_lhood;
  int8_t *dev_labels;
  cudaMalloc(&dev_samples, sz.samples_map);
  cudaMalloc(&dev_data, sz.data);
  cudaMalloc(&dev_labels, sz.labels);
  cudaMalloc(&dev_lhood, sz.likelihood);    // kernel will return a vector of likelihoods  

  // initialize likelihood to zeros (zero padding)
  memeset(host_lhood, 0, sz.likelihood * sizeof(double));
  // load data, labels, zero padded samples and likelihood on GPU
  cudaMemcpy(dev_data, data.gpudata, sz.data, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_labels, data.gpulabels, sz.labels, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_samples, mclocv.current, sz.samples_map, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_lhood, host_lhood, sz.likelihood, cudaMemcpyHostToDevice);  

  // tune_target_a_gpu(r, mcin, &mct, mclocv, mcloc, sz, 
  //                     dev_samples, dev_data, dev_labels,
  //                     dev_lhood, host_lhood);
  tune_target_a_gpu_v2(r, mcin, &mct, mclocv, mcloc, sz, 
                        dev_samples, dev_data, dev_labels,
                        dev_lhood, host_lhood);
  // tune_ess_gpu(r, mcin, &mct, mclocv, mcloc, sz, 
  //               dev_samples, dev_data, dev_labels,
  //               dev_lhood, host_lhood);

  burn_in_metropolis_gpu(r, mcin, mct, mcdata, mclocv, &mcloc, sz,
                          dev_samples, dev_data, dev_labels,
                          dev_lhood, host_lhood);
  
  int accepted_samples = 0;
  clock_t start, stop;  

  start  = clock();
  metropolis_gpu(r, mcin, mct, mcdata, mclocv, &mcloc, &accepted_samples, sz,
                  dev_samples, dev_data, dev_labels, dev_lhood, host_lhood);
  stop = clock() - start;
  
  out_par->time_m = stop / (CLOCKS_PER_SEC * 60);
  out_par->time_s = (stop / CLOCKS_PER_SEC) - (out_par->time_m * 60);
  out_par->time_ms = (stop * 1000 / CLOCKS_PER_SEC) - (out_par->time_s * 1000) 
                      - (out_par->time_m * 1000 * 60);
  out_par->acc_ratio = (double)accepted_samples / mcin.Ns;

  free_mcmc_vectors_gpu(mclocv);
}

void metropolis_gpu(gsl_rng *r, mcmc_str mcin,
                    mcmc_tune_str mct, mcmc_v_str mcdata, mcmc_int_v mclocv, 
                    mcmc_int *mcloc, int *accepted_samples, sz_str sz,
                    double *dev_samples, double *dev_data, double *dev_labels,
                    double *dev_lhood, double *host_lhood)
{
  int i, dim_idx;
  int lhood_idx;
  double plhood;

  fprintf(stdout, "Starting metropolis algorithm. Selected rwsd = %f\n", mct.rwsd); 
  
  for(i=0; i<mcin.Ns; i++)
  {
    // propose next sample (note that from mcin.ddata -> mcin.dmap is zero padded)
    for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
      mclocv.proposed[dim_idx] = mclocv.current[dim_idx] + gsl_ran_gaussian_ziggurat(r, mct.rwsd); // random walk using Marsaglia-Tsang ziggurat algorithm
    
    // load proposed samples on GPU
    cudaMemcpy(dev_samples, mclocv.proposed, sz.samples_actual, cudaMemcpyHostToDevice);
    // load kernel, calculate the proposed likelihood
    Likelihood_v1_sequential_addressing<<<mcin.Nd, mcin.dmap>>>>(dev_samples, dev_data, dev_labels, 
                                                                 dev_lhood);
    // return back a vector of likelihoods
    cudaMemcpy(host_lhood, dev_lhood, sz.likelihood, cudaMemcpyDeviceToHost);
    // finish single result on CPU
    plhood = 0;
    for(lhood_idx = 0; lhood_idx < mcin.Nd; lhood_idx++){
      plhood += host_lhood[lhood_idx];
    }

    // calculate acceptance ratio
    mcloc->acceptance = acceptance_ratio(mclocv, mcloc, mcin, plhood);
    
    mcloc->u = gsl_rng_uniform(r);

    if(mcloc->u <= mcloc->acceptance)
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

void burn_in_metropolis_gpu(gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata, 
                            mcmc_int_v mclocv, mcmc_int *mcloc, sz_str sz,
                            double *dev_samples, double *dev_data, double *dev_labels,
                            double *dev_lhood, double *host_lhood)
{
  int i, dim_idx, lhood_idx;
  double plhood, clhood;
  fprintf(stdout, "Starting burn in process. Selected rwsd = %f\n", mct.rwsd);

  // initialize burn in sequence
  for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
    mclocv.current[dim_idx] = mcdata.burn[dim_idx];

  // load current samples on GPU
  cudaMemcpy(dev_samples, mclocv.current, sz.samples_actual, cudaMemcpyHostToDevice);
  // load kernel, calculate the current likelihood
  Likelihood_v1_sequential_addressing<<<mcin.Nd, mcin.dmap>>>>(dev_samples, dev_data, dev_labels, 
                                                               dev_lhood);
  // return back a vector of likelihoods
  cudaMemcpy(host_lhood, dev_lhood, sz.likelihood, cudaMemcpyDeviceToHost);
  // finish single result on CPU
  clhood = 0;
  for(lhood_idx = 0; lhood_idx < mcin.Nd; lhood_idx++){
    clhood += host_lhood[lhood_idx];
  }

  // calculate the current posterior
  mcloc->cposterior = log_prior(mclocv.current, mcin) + clhood;

  // start burn in
  for(i=1; i<mcin.burnin; i++)
  {
    // propose next sample
    for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
      mclocv.proposed[dim_idx] = mclocv.current[dim_idx] + gsl_ran_gaussian_ziggurat(r, mct.rwsd); // random walk using Marsaglia-Tsang ziggurat algorithm
  
    // load proposed samples on GPU
    cudaMemcpy(dev_samples, mclocv.proposed, sz.samples_actual, cudaMemcpyHostToDevice);
    // load kernel, calculate the current likelihood
    Likelihood_v1_sequential_addressing<<<mcin.Nd, mcin.dmap>>>>(dev_samples, dev_data, dev_labels, 
                                                                 dev_lhood);
    // return back a vector of likelihoods
    cudaMemcpy(host_lhood, dev_lhood, sz.likelihood, cudaMemcpyDeviceToHost);
    // finish single result on CPU
    plhood = 0;
    for(lhood_idx = 0; lhood_idx < mcin.Nd; lhood_idx++){
      plhood += host_lhood[lhood_idx];
    }
    
    mcloc->acceptance = acceptance_ratio(mclocv, mcloc, mcin, plhood);
    mcloc->u = gsl_rng_uniform(r);

    if(mcloc->u <= mcloc->acceptance)
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
void tune_ess_gpu(gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata, 
                  mcmc_int_v mclocv, mcmc_int mcloc, sz_str sz,
                  double *dev_samples, double *dev_data, double *dev_labels,
                  double *dev_lhood, double *host_lhood)
{
  int chain_length = 5000;
  int runs = 40;
  double target_a[] = {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50};
  double error_tolerance = 0.01;
  double min_error = 9999999;
  double max_ess = -9999999;
  double lagidx = 500;

  double sd = mct->rwsd;

  int accepted_samples, run, a_idx;
  double acc_ratio_c, acc_error_c, best_acc_ratio;
  double circ_sum, best_sd, ess_sd, ess_c;

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

      short_run_burn_in(r, mclocv, mcin, sd, &mcloc, sz,
                          dev_samples, dev_data, dev_labels,
                          dev_lhood, host_lhood);
      short_run_metropolis(r, mclocv, mcin, chain_length, sd, &mcloc, 
                            samples, &accepted_samples, sz, dev_samples, 
                            dev_data, dev_labels, dev_lhood, host_lhood);
      
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
  free_mcmc_vectors_cpu(mclocv);
}


// tune rwsd for a target acceptance ratio
void tune_target_a_gpu_v2(gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata, 
                          mcmc_int_v mclocv, mcmc_int mcloc, sz_str sz,
                          double *dev_samples, double *dev_data, double *dev_labels,
                          double *dev_lhood, double *host_lhood)
{
  int chain_length = 5000;
  int runs = 40;
  double target_a = 0.25;
  double error_tolerance = 0.01;
  double min_error = 9999999;

  double sd = mct->rwsd;

  int accepted_samples, run;
  double acc_ratio_c, acc_error_c, best_acc_ratio, best_sd;

  double *samples = NULL;
  samples = (double*) malloc(mcin.ddata * chain_length * sizeof(double));
  if(samples == NULL)
    fprintf(stderr, "ERROR: Samples vector did not allocated.\n");

  fprintf(stdout, "\nStarting tuning process. Rwsd = %5.3f\n", sd);
  
  for(run=0; run<runs; run++)
  {
    fprintf(stdout, "\tStarting Run %2d. Current rwsd = %5.3f\n", run, sd);
    accepted_samples = 0;

    short_run_burn_in(r, mclocv, mcin, sd, &mcloc, sz,
                        dev_samples, dev_data, dev_labels,
                        dev_lhood, host_lhood);
    short_run_metropolis(r, mclocv, mcin, chain_length, sd, &mcloc, 
                          samples, &accepted_samples, sz, dev_samples, 
                          dev_data, dev_labels, dev_lhood, host_lhood);

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
  free_mcmc_vectors_cpu(mclocv);
}

// tune rwsd for a target acceptance ratio
void tune_target_a_gpu(gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata, 
                        mcmc_int_v mclocv, mcmc_int mcloc, sz_str sz,
                        double *dev_samples, double *dev_data, double *dev_labels,
                        double *dev_lhood, double *host_lhood)
{
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

    short_run_burn_in(r, mclocv, mcin, sd, &mcloc, sz,
                        dev_samples, dev_data, dev_labels,
                        dev_lhood, host_lhood);
    short_run_metropolis(r, mclocv, mcin, chain_length, sd, &mcloc, 
                          samples, &accepted_samples, sz, dev_samples, 
                          dev_data, dev_labels, dev_lhood, host_lhood);
    
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
  free_mcmc_vectors_cpu(mclocv);
}

void short_run_burn_in(gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, 
                        double sd, mcmc_int *mcloc, sz_str sz,
                        double *dev_samples, double *dev_data, double *dev_labels,
                        double *dev_lhood, double *host_lhood)
{
  int i, dim_idx, lhood_idx;
  double plhood, clhood;

  // initialize burn in sequence
  for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
    mclocv.current[dim_idx] = 0;
  
  // load current samples on GPU
  cudaMemcpy(dev_samples, mclocv.current, sz.samples_actual, cudaMemcpyHostToDevice);
  // load kernel, calculate the current likelihood
  Likelihood_v1_sequential_addressing<<<mcin.Nd, mcin.dmap>>>>(dev_samples, dev_data, dev_labels, 
                                                               dev_lhood);
  // return back a vector of likelihoods
  cudaMemcpy(host_lhood, dev_lhood, sz.likelihood, cudaMemcpyDeviceToHost);
  // finish single result on CPU
  clhood = 0;
  for(lhood_idx = 0; lhood_idx < mcin.Nd; lhood_idx++){
    clhood += host_lhood[lhood_idx];
  }

  // calculate the current posterior
  mcloc->cposterior = log_prior(mclocv.current, mcin) + clhood;  

  // start burn-in
  for(i=1; i<mcin.burnin; i++)
  {
    for(dim_idx = 0; dim_idx < mcin.ddata; dim_idx++){
      mclocv.proposed[dim_idx] = mclocv.current[dim_idx] 
                                  + gsl_ran_gaussian_ziggurat(r, sd); // random walk using Marsaglia-Tsang ziggurat algorithm  
    }

    // load proposed samples on GPU
    cudaMemcpy(dev_samples, mclocv.proposed, sz.samples_actual, cudaMemcpyHostToDevice);
    // load kernel, calculate the current likelihood
    Likelihood_v1_sequential_addressing<<<mcin.Nd, mcin.dmap>>>>(dev_samples, dev_data, dev_labels, 
                                                                 dev_lhood);
    // return back a vector of likelihoods
    cudaMemcpy(host_lhood, dev_lhood, sz.likelihood, cudaMemcpyDeviceToHost);
    // finish single result on CPU
    plhood = 0;
    for(lhood_idx = 0; lhood_idx < mcin.Nd; lhood_idx++){
      plhood += host_lhood[lhood_idx];
    }
    
    mcloc->acceptance = acceptance_ratio(mclocv, mcloc, mcin, plhood);
    mcloc->u = gsl_rng_uniform(r);

    if(mcloc->u <= mcloc->acceptance)    // decide if you accept the proposed theta or not
    {
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++){
        mclocv.current[dim_idx] = mclocv.proposed[dim_idx];
      }
      mcloc->cposterior = mcloc->pposterior; // make proposed posterior the current 
    }
  }
}

void short_run_metropolis(data_str data, gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, 
                          int chain_length, double sd, mcmc_int *mcloc, double *samples, 
                          int *accepted_samples, sz_str sz, double *dev_samples, 
                          double *dev_data, double *dev_labels,
                          double *dev_lhood, double *host_lhood)
{
  int i, dim_idx;
  int lhood_idx;
  double plhood;
  
  // start metropolis
  for(i=0; i < chain_length; i++){    
    for(dim_idx = 0; dim_idx < mcin.ddata; dim_idx++){
      mclocv.proposed[dim_idx] = mclocv.current[dim_idx] 
                                  + gsl_ran_gaussian_ziggurat(r, sd); // random walk using Marsaglia-Tsang ziggurat algorithm    
    }

    // load proposed samples on GPU
    cudaMemcpy(dev_samples, mclocv.proposed, sz.samples_actual, cudaMemcpyHostToDevice);
    // load kernel, calculate the proposed likelihood
    Likelihood_v1_sequential_addressing<<<mcin.Nd, mcin.dmap>>>>(dev_samples, dev_data, dev_labels, 
                                                                 dev_lhood);
    // return back a vector of likelihoods
    cudaMemcpy(host_lhood, dev_lhood, sz.likelihood, cudaMemcpyDeviceToHost);
    // finish single result on CPU
    plhood = 0;
    for(lhood_idx = 0; lhood_idx < mcin.Nd; lhood_idx++){
      plhood += host_lhood[lhood_idx];
    }

    // calculate acceptance ratio
    mcloc->acceptance = acceptance_ratio(mclocv, mcloc, mcin, plhood);
    mcloc->u = gsl_rng_uniform(r);

    if(mcloc->u <= mcloc->acceptance)    // decide if you accept the proposed theta or not
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

double acceptance_ratio(mcmc_int_v mclocv, mcmc_int *mcloc, mcmc_str mcin, double plhood) 
{
  double log_ratio;
  mcloc->pposterior = log_prior(mclocv.proposed, mcin) + plhood;
  log_ratio = mcloc->pposterior - mcloc->cposterior;

  return exp(log_ratio);
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

void print_gpu_info()
{
  cudaDeviceProp  prop;
  int count;
  cudaGetDeviceCount( &count );

  for (int i=0; i< count; i++) {
    cudaGetDeviceProperties( &prop, i );
    //Do something with our device's properties
    printf( " --- General Information for device %d ---\n", i ); 
    printf( "Name: %s\n", prop.name );
    printf( "Compute capability: %d.%d\n", prop.major, prop.minor ); 
    printf( "Clock rate: %d\n", prop.clockRate );
    printf( "Device copy overlap: " );
    if (prop.deviceOverlap)
      printf( "Enabled\n" ); 
    else
      printf( "Disabled\n" );

    printf( "Kernel execition timeout : " ); 
    if (prop.kernelExecTimeoutEnabled)
      printf( "Enabled\n" ); 
    else
      printf( "Disabled\n" );

    printf( "   --- Memory Information for device %d ---\n", i );
    printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
    printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
    printf( "Max mem pitch:  %ld\n", prop.memPitch );
    printf( "Texture Alignment:  %ld\n", prop.textureAlignment );
    printf( "   --- MP Information for device %d ---\n", i );
    printf( "Multiprocessor count:  %d\n",
             prop.multiProcessorCount );
    printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
    printf( "Registers per mp:  %d\n", prop.regsPerBlock );
    printf( "Threads in warp:  %d\n", prop.warpSize );
    printf( "Max threads per block:  %d\n",
               prop.maxThreadsPerBlock );
    printf( "Max thread dimensions:  (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1],
               prop.maxThreadsDim[2] );
    printf( "Max grid dimensions:  (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1],
               prop.maxGridSize[2] );
    printf( "\n" );
  }  
}

/****************************************************************************/
void Metropolis_Hastings_gpu(data_str data, gsl_rng *r, mcmc_str mcin,
                              mcmc_tune_str mct, mcmc_v_str mcdata,
                              out_str *out_par)
{
  print_gpu_info();
  cudaSetDevice(0);

  mcmc_int_v mclocv;
  mcmc_int mcloc;
  malloc_mcmc_vectors_gpu(&mclocv, mcin);

  int accepted_samples = 0;
  int dim_idx,i;

  int threadsPerBlock = mcin.dmap;
  int blocksPerGrid = mcin.Nd;

  sz_str sz;
  sz.samples = mcin.dmap * sizeof(double);
  sz.data = mcin.dmap * mcin.Nd * sizeof(double);
  sz.labels = mcin.Nd * sizeof(int8_t);
  sz.likelihood = blocksPerGrid * sizeof(double);

  double *host_lhood = (double *) malloc(sz.likelihood * sizeof(double));
  double *dev_samples, *dev_data, *dev_lhood;
  int8_t *dev_labels;
  cudaMalloc(&dev_samples, sz.samples);
  cudaMalloc(&dev_data, sz.data);
  cudaMalloc(&dev_labels, sz.labels);
  cudaMalloc(&dev_lhood, sz.likelihood);    // kernel will return a vector of likelihoods

  clock_t start, stop;
  start  = clock();

  // initialisation
  memcpy(mclocv.current, mcdata.burn, mcin.ddata*sizeof(double));

  // load data, labels and current samples on gpu
  cudaMemcpy(dev_data, data.gpudata, sz.data, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_labels, data.labels, sz.labels, cudaMemcpyHostToDevice);
  
  cudaMemcpy(dev_samples, mclocv.current, sz.samples, cudaMemcpyHostToDevice);
  Likelihood_v1<<<blocksPerGrid, threadsPerBlock>>>(dev_samples, dev_data, dev_labels, 
                                                    mcin.ddata, mcin.dmap, mcin.Nd, 
                                                    dev_lhood);
  cudaMemcpy(host_lhood, dev_lhood, sz.likelihood, cudaMemcpyDeviceToHost);
  // finish on cpu
  uint like_idx;
  double cpu_likelihood = 0;
  for(like_idx = 0; like_idx < sz.likelihood; like_idx++){
    cpu_likelihood += host_lhood[like_idx];
  }

  mcloc.cposterior = log_prior(mclocv.current, mcin) + cpu_likelihood;
  
  //perfrom metropolis-hastings algorithm
  for(i=1; i<(mcin.Ns+mcin.burnin); i++){  
    for(dim_idx = 0; dim_idx < mcin.ddata; dim_idx++){
      mclocv.proposed[dim_idx] = mclocv.current[dim_idx] 
                                  + gsl_ran_gaussian_ziggurat(r, mct.rwsd); // random walk using Marsaglia-Tsang ziggurat algorithm
    }

    cudaMemcpy(dev_samples, mclocv.current, sz.samples, cudaMemcpyHostToDevice);
    Likelihood_v1<<<blocksPerGrid, threadsPerBlock>>>(dev_samples, dev_data, dev_labels, 
                                                      mcin.ddata, mcin.dmap, mcin.Nd, 
                                                      dev_lhood);
    cudaMemcpy(host_lhood, dev_lhood, sz.likelihood, cudaMemcpyDeviceToHost);
    // finish on cpu
    cpu_likelihood = 0;
    for(like_idx = 0; like_idx < sz.likelihood; like_idx++){
      cpu_likelihood += host_lhood[like_idx];
    }

    mcloc.acceptance = acceptance_ratio(mclocv, &mcloc, mcin, cpu_likelihood);// Calculate acceptance ratio in the log domain
    mcloc.u = gsl_rng_uniform(r);

    if(mcloc.u <= mcloc.acceptance)    // decide if you accept the proposed theta or not
    {
      memcpy(mclocv.temp, mclocv.proposed, mcin.ddata*sizeof(double));
      memcpy(mclocv.current, mclocv.proposed, mcin.dmap*sizeof(double));
      mcloc.cposterior = mcloc.pposterior; // make proposed posterior the current
      accepted_samples += 1;   
    }else{
      memcpy(mclocv.temp, mclocv.current, mcin.ddata*sizeof(double));
    }

    if(i < mcin.burnin){
      for(dim_idx = 0; dim_idx < mcin.ddata; dim_idx++)
        mcdata.burn[(i * mcin.ddata) + dim_idx] = mclocv.temp[dim_idx];
    }else{
      for(dim_idx = 0; dim_idx < mcin.ddata; dim_idx++)
        mcdata.samples[(i - mcin.burnin) * mcin.ddata + dim_idx] = mclocv.temp[dim_idx];
    }        
  }

  stop = clock() - start;
  out_par->time_m = stop / (CLOCKS_PER_SEC * 60);
  out_par->time_s = (stop / CLOCKS_PER_SEC) - (out_par->time_m * 60);
  out_par->time_ms = (stop * 1000 / CLOCKS_PER_SEC) - (out_par->time_s * 1000) 
                      - (out_par->time_m * 1000 * 60);

  out_par -> acc_ratio = (double)accepted_samples / mcin.Ns;

  free_mcmc_vectors_gpu(mclocv);
  free(host_lhood);
}


