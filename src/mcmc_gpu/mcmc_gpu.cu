#ifndef __MCMC_GPU_CU__
#define __MCMC_GPU_CU__

#include "mcmc_gpu.h"

const int PRIOR_SD = 10;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void gpu_sampler(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, 
                  mcmc_v_str mcdata, gpu_v_str gpu, out_str *res)
{
  int accepted_samples;

  clock_t startBurn, stopBurn;
  clock_t startMcmc, stopMcmc;
  // print_gpu_info();
  cudaSetDevice(0);

  cublasHandle_t handle;
  cublasCreate(&handle);

  mcmc_int_v mclocv;
  mcmc_int mcloc;
  mcloc.cposterior = 0;
  mcloc.pposterior = 0;
  mcloc.acceptance = 0;
  mcloc.u = 0;
  malloc_mcmc_vectors(&mclocv, mcin);

  // set up the gpu vectors
  sz_str sz;
  dev_v_str d;

  getBlocksAndThreads(gpu.kernel, mcin.Nd, gpu.maxBlocks, gpu.maxThreads, &gpu.blocks, &gpu.threads);

  sz.samples = mcin.ddata * sizeof(double);
  sz.data = mcin.ddata * mcin.Nd * sizeof(double);
  sz.cuLhood = mcin.Nd * sizeof(double);
  sz.lhood = mcin.Nd * sizeof(double);

  double *host_lhood = (double*) malloc(mcin.Nd*sizeof(double));
  cudaMalloc(&d.samples, sz.samples);
  cudaMalloc(&d.data, sz.data);
  cudaMalloc(&d.cuLhood, sz.cuLhood);    // kernel will return a vector of likelihoods  
  cudaMalloc(&d.lhood, sz.lhood);

  cudaMemcpy(d.data, data.data, sz.data, cudaMemcpyHostToDevice);

  startBurn = clock();
  if(mcin.burnin != 0)
    burn_in_metropolis_gpu(handle, r, mcin, mct, mcdata, mclocv, &mcloc, sz, gpu, d, host_lhood);
  stopBurn = clock() - startBurn;

  accepted_samples = 0;  

  startMcmc = clock();

  metropolis_gpu(handle, r, mcin, mct, mcdata, mclocv, &mcloc, &accepted_samples, sz, gpu, d, host_lhood, res);

  stopMcmc = clock() - startMcmc;

  res->burnTime = stopBurn * 1000 / CLOCKS_PER_SEC;   // burn in time in ms
  res->mcmcTime = stopMcmc * 1000 / CLOCKS_PER_SEC;   // mcmc time in ms
  res->acceptance = (double)accepted_samples / mcin.Ns;
  
  cublasDestroy(handle);
  cudaFree(d.samples);
  cudaFree(d.data);
  cudaFree(d.cuLhood);
  cudaFree(d.lhood);
  free(host_lhood);
  free_mcmc_vectors(mclocv, mcin);
}

void metropolis_gpu(cublasHandle_t handle, gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata, 
                    mcmc_int_v mclocv, mcmc_int *mcloc, int *accepted_samples, sz_str sz, 
                    gpu_v_str gpu, dev_v_str d, double *host_lhood, out_str *res)
{
  int i, dim_idx;
  double plhood;
  res->cuTime = 0;
  res->cuBandwidth = 0;
  res->kernelTime = 0;
  res->kernelBandwidth = 0;
  res->gpuTime = 0;
  res->gpuBandwidth = 0;

  fprintf(stdout, "Starting metropolis algorithm. Selected rwsd = %f\n", mct.rwsd); 
  
  for(i=0; i<mcin.Ns; i++)
  {
    for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
      // random walk using Marsaglia-Tsang ziggurat algorithm
      mclocv.proposed[dim_idx] = mclocv.current[dim_idx] + gsl_ran_gaussian_ziggurat(r, mct.rwsd);

    plhood = gpu_likelihood_d(handle, mcin, gpu, mclocv.proposed, sz.samples, d, host_lhood, res);
    
    // calculate acceptance ratio
    mcloc->acceptance = acceptance_ratio_gpu(mclocv, mcloc, mcin, plhood);
    
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

void burn_in_metropolis_gpu(cublasHandle_t handle, gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata, 
                              mcmc_int_v mclocv, mcmc_int *mcloc, sz_str sz, gpu_v_str gpu, 
                              dev_v_str d, double *host_lhood)
{
  int i, dim_idx;
  double plhood, clhood;
  out_str res;

  fprintf(stdout, "Starting burn in process. Selected rwsd = %f\n", mct.rwsd);
  // initialize burn in sequence
  for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
    mclocv.current[dim_idx] = mcdata.burn[dim_idx];

  clhood = gpu_likelihood_d(handle, mcin, gpu, mclocv.current, sz.samples, d, host_lhood, &res);
  // calculate the current posterior
  mcloc->cposterior = log_prior_gpu(mclocv.current, mcin) + clhood;

  // start burn in
  for(i=1; i<mcin.burnin; i++)
  {
    // propose next sample
    for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
      mclocv.proposed[dim_idx] = mclocv.current[dim_idx] + gsl_ran_gaussian_ziggurat(r, mct.rwsd); // random walk using Marsaglia-Tsang ziggurat algorithm

    plhood = gpu_likelihood_d(handle, mcin, gpu, mclocv.proposed, sz.samples, d, host_lhood, &res);

    mcloc->acceptance = acceptance_ratio_gpu(mclocv, mcloc, mcin, plhood);
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

double reduction_d(gpu_v_str gpu, dev_v_str d, double *host_lhood, double *ke_acc_Bytes)
{
  double gpu_result = 0;
  int i;
  int numBlocks = gpu.blocks;
  int threads, blocks;

  *ke_acc_Bytes = gpu.size * sizeof(double);

  reduceSum_d(gpu.size, gpu.threads, gpu.blocks, gpu.kernel, d.cuLhood, d.lhood, LogRegression);

  while(numBlocks >= gpu.cpuThresh)
  {
    getBlocksAndThreads(gpu.kernel, numBlocks, gpu.maxBlocks, gpu.maxThreads, &blocks, &threads);
    
    ke_acc_Bytes += numBlocks * sizeof(double);
    
    gpuErrchk(cudaMemcpy(d.cuLhood, d.lhood, numBlocks*sizeof(double), cudaMemcpyDeviceToDevice));
    reduceSum_d(numBlocks, threads, blocks, gpu.kernel, d.cuLhood, d.lhood, Reduction);    

    if(gpu.kernel < 3)
    {
      numBlocks = (numBlocks + threads - 1) / threads;
    }else{
      numBlocks = (numBlocks +(threads*2-1)) / (threads*2);
    }
  }

  gpuErrchk(cudaMemcpy(host_lhood, d.lhood, numBlocks*sizeof(double), cudaMemcpyDeviceToHost));  
  // accumulate result on CPU
  for(i=0; i<numBlocks; i++){
    gpu_result += host_lhood[i];
  }

  return gpu_result;
}

double gpu_likelihood_d(cublasHandle_t handle, mcmc_str mcin, gpu_v_str gpu, double *samples, size_t sampleSz, 
                        dev_v_str d, double *host_lhood, out_str *res)
{
  double ke_acc_Bytes = 0;
  double cuBytes = 0;
  double reduced_lhood = 0;
  double a = 1.0;
  double b = 0.0;
  float cu_ms = 0;
  float ke_ms = 0;

  cudaEvent_t cuStart, cuStop, keStart, keStop;
  cudaEventCreate(&cuStart); 
  cudaEventCreate(&cuStop);
  cudaEventCreate(&keStart);
  cudaEventCreate(&keStop);  

  gpuErrchk(cudaMemcpy(d.samples, samples, sampleSz, cudaMemcpyHostToDevice));
  cudaEventRecord(cuStart);  
  cublasDgemv(handle, CUBLAS_OP_N, mcin.Nd, mcin.ddata, &a, d.data, mcin.Nd, d.samples, 1, &b, d.cuLhood, 1);
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(cuStop);

  cudaEventRecord(keStart);
  getBlocksAndThreads(gpu.kernel, mcin.Nd, gpu.maxBlocks, gpu.maxThreads, &gpu.blocks, &gpu.threads);
  reduced_lhood = reduction_d(gpu, d, host_lhood, &ke_acc_Bytes);
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(keStop);

  cudaEventSynchronize(cuStop); 
  cudaEventSynchronize(keStop);
  cudaEventElapsedTime(&cu_ms, cuStart, cuStop);
  cudaEventElapsedTime(&ke_ms, keStart, keStop);

  cuBytes = mcin.Nd * (mcin.ddata + 2) * sizeof(double);
  res->cuTime += cu_ms / mcin.Ns;    // average cuBlas time
  res->cuBandwidth += (cuBytes / cu_ms / 1e6) / mcin.Ns;
  res->kernelTime += ke_ms / mcin.Ns;
  res->kernelBandwidth += (ke_acc_Bytes / ke_ms / 1e6) / mcin.Ns;
  res->gpuTime += (cu_ms + ke_ms) / mcin.Ns;
  res->gpuBandwidth += ((cuBytes + ke_acc_Bytes) / (cu_ms + ke_ms) / 1e6) / mcin.Ns;
// printf("ok8\n");
  return reduced_lhood;  
}

// // tune rwsd for a target acceptance ratio
void tune_ess_gpu(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, gpu_v_str gpu, double *initCond, int length)
{
  // print_gpu_info();
  cudaSetDevice(0);

  cublasHandle_t handle;
  cublasCreate(&handle);

  mcmc_int_v mclocv;
  mcmc_int mcloc;
  mcloc.cposterior = 0;
  mcloc.pposterior = 0;
  mcloc.acceptance = 0;
  mcloc.u = 0;
  malloc_mcmc_vectors(&mclocv, mcin);

  // set up the gpu vectors
  sz_str sz;
  dev_v_str d;

  getBlocksAndThreads(gpu.kernel, mcin.Nd, gpu.maxBlocks, gpu.maxThreads, &gpu.blocks, &gpu.threads);

  sz.samples = mcin.ddata * sizeof(double);
  sz.data = mcin.ddata * mcin.Nd * sizeof(double);
  sz.cuLhood = mcin.Nd * sizeof(double);
  sz.lhood = gpu.blocks * sizeof(double);

  cudaMalloc(&d.samples, sz.samples);
  cudaMalloc(&d.data, sz.data);
  cudaMalloc(&d.cuLhood, sz.cuLhood);    // kernel will return a vector of likelihoods  
  cudaMalloc(&d.lhood, sz.lhood);

  cudaMemcpy(d.data, data.data, sz.data, cudaMemcpyHostToDevice);  

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

      short_run_burn_in_gpu(handle, r, mclocv, mcin, sd, &mcloc, sz, gpu, d, data.mvout, initCond);
      short_run_metropolis_gpu(handle, r, mclocv, mcin, chain_length, sd, &mcloc, 
                                samples, &accepted_samples, sz, gpu, d, data.mvout);
      
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
  
  cublasDestroy(handle);
  free(samples);
  free(autocorr_lagk);
  cudaFree(d.samples);
  cudaFree(d.data);
  cudaFree(d.cuLhood);
  cudaFree(d.lhood);
  free_mcmc_vectors(mclocv, mcin);
}


// // tune rwsd for a target acceptance ratio
void tune_target_a_gpu_v2(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, 
                          gpu_v_str gpu, double *initCond, double ratio, int max_reps)
{
  // print_gpu_info();
  cudaSetDevice(0);

  cublasHandle_t handle;
  cublasCreate(&handle);

  mcmc_int_v mclocv;
  mcmc_int mcloc;
  mcloc.cposterior = 0;
  mcloc.pposterior = 0;
  mcloc.acceptance = 0;
  mcloc.u = 0;
  malloc_mcmc_vectors(&mclocv, mcin);

  // set up the gpu vectors
  sz_str sz;
  dev_v_str d;

  getBlocksAndThreads(gpu.kernel, mcin.Nd, gpu.maxBlocks, gpu.maxThreads, &gpu.blocks, &gpu.threads);

  sz.samples = mcin.ddata * sizeof(double);
  sz.data = mcin.ddata * mcin.Nd * sizeof(double);
  sz.cuLhood = mcin.Nd * sizeof(double);
  sz.lhood = gpu.blocks * sizeof(double);

  cudaMalloc(&d.samples, sz.samples);
  cudaMalloc(&d.data, sz.data);
  cudaMalloc(&d.cuLhood, sz.cuLhood);    // kernel will return a vector of likelihoods  
  cudaMalloc(&d.lhood, sz.lhood);

  cudaMemcpy(d.data, data.data, sz.data, cudaMemcpyHostToDevice);

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

    short_run_burn_in_gpu(handle, r, mclocv, mcin, sd, &mcloc, sz, gpu, d, data.mvout, initCond);
    short_run_metropolis_gpu(handle, r, mclocv, mcin, chain_length, sd, &mcloc, 
                              samples, &accepted_samples, sz, gpu, d, data.mvout);

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
  
  cublasDestroy(handle);
  free(samples);
  cudaFree(d.samples);
  cudaFree(d.data);
  cudaFree(d.cuLhood);
  cudaFree(d.lhood);
  free_mcmc_vectors(mclocv, mcin);
}

// tune rwsd for a target acceptance ratio
void tune_target_a_gpu(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, gpu_v_str gpu, double *initCond, int length)
{
  // print_gpu_info();
  cudaSetDevice(0);

  cublasHandle_t handle;
  cublasCreate(&handle);

  mcmc_int_v mclocv;
  mcmc_int mcloc;
  mcloc.cposterior = 0;
  mcloc.pposterior = 0;
  mcloc.acceptance = 0;
  mcloc.u = 0;
  malloc_mcmc_vectors(&mclocv, mcin);

  // set up the gpu vectors
  sz_str sz;
  dev_v_str d;

  getBlocksAndThreads(gpu.kernel, mcin.Nd, gpu.maxBlocks, gpu.maxThreads, &gpu.blocks, &gpu.threads);

  sz.samples = mcin.ddata * sizeof(double);
  sz.data = mcin.ddata * mcin.Nd * sizeof(double);
  sz.cuLhood = mcin.Nd * sizeof(double);
  sz.lhood = gpu.blocks * sizeof(double);

  cudaMalloc(&d.samples, sz.samples);
  cudaMalloc(&d.data, sz.data);
  cudaMalloc(&d.cuLhood, sz.cuLhood);    // kernel will return a vector of likelihoods  
  cudaMalloc(&d.lhood, sz.lhood);

  cudaMemcpy(d.data, data.data, sz.data, cudaMemcpyHostToDevice);

  int chain_length = length;
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

    short_run_burn_in_gpu(handle, r, mclocv, mcin, sd, &mcloc, sz, gpu, d, data.mvout, initCond);
    short_run_metropolis_gpu(handle, r, mclocv, mcin, chain_length, sd, &mcloc, 
                              samples, &accepted_samples, sz, gpu, d, data.mvout);
    
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

  cublasDestroy(handle);  
  free(samples);
  cudaFree(d.samples);
  cudaFree(d.data);
  cudaFree(d.cuLhood);
  cudaFree(d.lhood);
  free_mcmc_vectors(mclocv, mcin);
}

void short_run_burn_in_gpu(cublasHandle_t handle, gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, double sd, mcmc_int *mcloc, 
                            sz_str sz, gpu_v_str gpu, dev_v_str d, double *host_lhood, double *initCond)
{
  int i, dim_idx;
  double plhood, clhood;

  out_str res;

  // initialize burn in sequence
  for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
    mclocv.current[dim_idx] = 0;

  clhood = gpu_likelihood_d(handle, mcin, gpu, mclocv.current, sz.samples, d, host_lhood, &res); 
  // calculate the current posterior
  mcloc->cposterior = log_prior_gpu(mclocv.current, mcin) + clhood;

  // start burn-in
  for(i=1; i<mcin.burnin; i++)
  {
    for(dim_idx = 0; dim_idx < mcin.ddata; dim_idx++){
      mclocv.proposed[dim_idx] = mclocv.current[dim_idx] 
                                  + gsl_ran_gaussian_ziggurat(r, sd); // random walk using Marsaglia-Tsang ziggurat algorithm  
    }

    plhood = gpu_likelihood_d(handle, mcin, gpu, mclocv.proposed, sz.samples, d, host_lhood, &res);

    mcloc->acceptance = acceptance_ratio_gpu(mclocv, mcloc, mcin, plhood);
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

void short_run_metropolis_gpu(cublasHandle_t handle, gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, int chain_length, double sd, 
                              mcmc_int *mcloc, double *samples, int *accepted_samples, sz_str sz, 
                              gpu_v_str gpu, dev_v_str d, double *host_lhood)
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

    plhood = gpu_likelihood_d(handle, mcin, gpu, mclocv.proposed, sz.samples, d, host_lhood, &res);

    mcloc->acceptance = acceptance_ratio_gpu(mclocv, mcloc, mcin, plhood);
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

double acceptance_ratio_gpu(mcmc_int_v mclocv, mcmc_int *mcloc, mcmc_str mcin, double plhood) 
{
  double log_ratio;
  mcloc->pposterior = log_prior_gpu(mclocv.proposed, mcin) + plhood;
  log_ratio = mcloc->pposterior - mcloc->cposterior;

  return log_ratio;
}

double log_prior_gpu(double *sample, mcmc_str mcin)
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
  int i;
  for (i=0; i< count; i++) {
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

    printf( "Kernel execution timeout : " ); 
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

#endif // __MCMC_GPU_CU__