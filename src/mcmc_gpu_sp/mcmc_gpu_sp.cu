#ifndef __MCMC_GPU_SP_CU__
#define __MCMC_GPU_SP_CU__

#include "mcmc_gpu_sp.h"

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


void sp_sampler(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, 
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
  mcloc.cposteriorf = 0;
  mcloc.pposteriorf = 0;
  mcloc.acceptancef = 0;
  mcloc.uf = 0;
  malloc_mcmc_vectors_sp(&mclocv, mcin);

  // set up the gpu vectors
  sz_str sz;
  dev_v_str d;

  getBlocksAndThreads(gpu.kernel, mcin.Nd, gpu.maxBlocks, gpu.maxThreads, &gpu.blocks, &gpu.threads);

  sz.samples = mcin.ddata * sizeof(float);
  sz.data = mcin.ddata * mcin.Nd * sizeof(float);
  sz.cuLhood = mcin.Nd * sizeof(float);
  sz.lhood = mcin.Nd * sizeof(float);

  float *host_lhoodf = (float*) malloc(mcin.Nd*sizeof(float));
  cudaMalloc(&d.samplesf, sz.samples);
  cudaMalloc(&d.dataf, sz.data);
  cudaMalloc(&d.cuLhoodf, sz.cuLhood);    // kernel will return a vector of likelihoods  
  cudaMalloc(&d.lhoodf, sz.lhood);

  cudaMemcpy(d.dataf, data.dataf, sz.data, cudaMemcpyHostToDevice);

  startBurn = clock();
  if(mcin.burnin != 0)
    burn_in_metropolis_sp(handle, r, mcin, mct, mcdata, mclocv, &mcloc, sz, gpu, d, host_lhoodf);
  stopBurn = clock() - startBurn;

  accepted_samples = 0;  

  startMcmc = clock();

  metropolis_sp(handle, r, mcin, mct, mcdata, mclocv, &mcloc, &accepted_samples, sz, gpu, d, host_lhoodf, res);

  stopMcmc = clock() - startMcmc;

  res->burnTime = stopBurn * 1000 / CLOCKS_PER_SEC;   // burn in time in ms
  res->mcmcTime = stopMcmc * 1000 / CLOCKS_PER_SEC;   // mcmc time in ms
  res->acceptance = (float)accepted_samples / mcin.Ns;
  
  cublasDestroy(handle);
  cudaFree(d.samples);
  cudaFree(d.data);
  cudaFree(d.cuLhood);
  cudaFree(d.lhood);
  free(host_lhoodf);
  free_mcmc_vectors_sp(mclocv, mcin);
}

void metropolis_sp(cublasHandle_t handle, gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata, 
                    mcmc_int_v mclocv, mcmc_int *mcloc, int *accepted_samples, sz_str sz, 
                    gpu_v_str gpu, dev_v_str d, float *host_lhoodf, out_str *res)
{
  int i, dim_idx;
  float plhood;
  res->cuTime = 0;
  res->cuBandwidth = 0;
  res->kernelTime = 0;
  res->kernelBandwidth = 0;
  res->gpuTime = 0;
  res->gpuBandwidth = 0;

  fprintf(stdout, "Starting metropolis algorithm. Selected rwsdf = %f\n", mct.rwsdf); 
  
  for(i=0; i<mcin.Ns; i++)
  {
    for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
      // random walk using Marsaglia-Tsang ziggurat algorithm
      mclocv.proposedf[dim_idx] = mclocv.currentf[dim_idx] + (float) gsl_ran_gaussian_ziggurat(r, (double)mct.rwsdf);

    plhood = gpu_likelihood_f(handle, mcin, gpu, mclocv.proposedf, sz.samples, d, host_lhoodf, res);
    
    // calculate acceptance ratio
    mcloc->acceptancef = acceptance_ratio_sp(mclocv, mcloc, mcin, plhood);
    
    mcloc->uf = gsl_rng_uniform(r);

    if(log(mcloc->uf) <= mcloc->acceptancef)
    {
      // accept proposed sample
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
      {
        mcdata.samplesf[i*mcin.ddata + dim_idx] = mclocv.proposedf[dim_idx];
        mclocv.currentf[dim_idx] = mclocv.proposedf[dim_idx];
      }
      mcloc->cposteriorf = mcloc->pposteriorf;
      *accepted_samples += 1;
    }else{
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
        mcdata.samplesf[i*mcin.ddata + dim_idx] = mclocv.currentf[dim_idx];
    }    
  } 
  fprintf(stdout, "Metropolis algorithm finished. Accepted Samples = %d\n\n", *accepted_samples);
}

void burn_in_metropolis_sp(cublasHandle_t handle, gsl_rng *r, mcmc_str mcin, mcmc_tune_str mct, mcmc_v_str mcdata, 
                              mcmc_int_v mclocv, mcmc_int *mcloc, sz_str sz, gpu_v_str gpu, 
                              dev_v_str d, float *host_lhoodf)
{
  int i, dim_idx;
  float plhood, clhood;
  out_str res;

  fprintf(stdout, "Starting burn in process. Selected rwsdf = %f\n", mct.rwsdf);
  // initialize burn in sequence
  for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
    mclocv.currentf[dim_idx] = mcdata.burnf[dim_idx];

  clhood = gpu_likelihood_f(handle, mcin, gpu, mclocv.currentf, sz.samples, d, host_lhoodf, &res);
  // calculate the current posterior
  mcloc->cposteriorf = log_prior_sp(mclocv.currentf, mcin) + clhood;

  // start burn in
  for(i=1; i<mcin.burnin; i++)
  {
    // propose next sample
    for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
      mclocv.proposedf[dim_idx] = mclocv.currentf[dim_idx] + (float) gsl_ran_gaussian_ziggurat(r, (double)mct.rwsdf); // random walk using Marsaglia-Tsang ziggurat algorithm
    
    plhood = gpu_likelihood_f(handle, mcin, gpu, mclocv.proposedf, sz.samples, d, host_lhoodf, &res);
    
    mcloc->acceptancef = acceptance_ratio_sp(mclocv, mcloc, mcin, plhood);
    mcloc->uf = gsl_rng_uniform(r);
    
    if(log(mcloc->uf) <= mcloc->acceptancef)
    {
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
      {
        mcdata.burnf[i*mcin.ddata + dim_idx] = mclocv.proposedf[dim_idx];
        mclocv.currentf[dim_idx] = mclocv.proposedf[dim_idx];
      }
      mcloc->cposteriorf = mcloc->pposteriorf;
// printf("ok10\n");
    }else{
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
        mcdata.burnf[i*mcin.ddata + dim_idx] = mclocv.currentf[dim_idx];
      // printf("ok11\n");
    }
  }
  fprintf(stdout, "Burn in process finished.\n\n");
}

float reduction_f(gpu_v_str gpu, dev_v_str d, float *host_lhoodf, float *ke_acc_Bytes)
{
  float gpu_result = 0;
  int i;
  int numBlocks = gpu.blocks;
  int threads, blocks;

  *ke_acc_Bytes = gpu.size * sizeof(float);

  reduceSum_f(gpu.size, gpu.threads, gpu.blocks, gpu.kernel, d.cuLhoodf, d.lhoodf, LogRegression);

  while(numBlocks >= gpu.cpuThresh)
  {
    getBlocksAndThreads(gpu.kernel, numBlocks, gpu.maxBlocks, gpu.maxThreads, &blocks, &threads);
    
    ke_acc_Bytes += numBlocks * sizeof(float);
    
    gpuErrchk(cudaMemcpy(d.cuLhoodf, d.lhoodf, numBlocks*sizeof(float), cudaMemcpyDeviceToDevice));
    reduceSum_f(numBlocks, threads, blocks, gpu.kernel, d.cuLhoodf, d.lhoodf, Reduction);    

    if(gpu.kernel < 3)
    {
      numBlocks = (numBlocks + threads - 1) / threads;
    }else{
      numBlocks = (numBlocks +(threads*2-1)) / (threads*2);
    }
  }

  gpuErrchk(cudaMemcpy(host_lhoodf, d.lhoodf, numBlocks*sizeof(float), cudaMemcpyDeviceToHost));  
  // accumulate result on CPU
  for(i=0; i<numBlocks; i++){
    gpu_result += host_lhoodf[i];
  }

  return gpu_result;
}

float gpu_likelihood_f(cublasHandle_t handle, mcmc_str mcin, gpu_v_str gpu, float *samplesf, float sampleSz, 
                        dev_v_str d, float *host_lhoodf, out_str *res)
{
  float ke_acc_Bytes = 0;
  float cuBytes = 0;
  float reduced_lhood = 0;
  float a = 1.0;
  float b = 0.0;
  float cu_ms = 0;
  float ke_ms = 0;


  cudaEvent_t cuStart, cuStop, keStart, keStop;
  gpuErrchk(cudaEventCreate(&cuStart)); 
  gpuErrchk(cudaEventCreate(&cuStop));
  gpuErrchk(cudaEventCreate(&keStart));
  gpuErrchk(cudaEventCreate(&keStop));  

  gpuErrchk(cudaMemcpy(d.samplesf, samplesf, sampleSz, cudaMemcpyHostToDevice));

  cudaEventRecord(cuStart);  
  cublasSgemv(handle, CUBLAS_OP_N, mcin.Nd, mcin.ddata, &a, d.dataf, mcin.Nd, d.samplesf, 1, &b, d.cuLhoodf, 1);
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(cuStop);

  cudaEventRecord(keStart);
  getBlocksAndThreads(gpu.kernel, mcin.Nd, gpu.maxBlocks, gpu.maxThreads, &gpu.blocks, &gpu.threads);

  reduced_lhood = reduction_f(gpu, d, host_lhoodf, &ke_acc_Bytes);

  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(keStop);

  cudaEventSynchronize(cuStop); 
  cudaEventSynchronize(keStop);
  cudaEventElapsedTime(&cu_ms, cuStart, cuStop);
  cudaEventElapsedTime(&ke_ms, keStart, keStop);

  cuBytes = mcin.Nd * (mcin.ddata + 2) * sizeof(float);

  res->cuTime += cu_ms / mcin.Ns;    // average cuBlas time
  res->cuBandwidth += (cuBytes / cu_ms / 1e6) / mcin.Ns;
  res->kernelTime += ke_ms / mcin.Ns;
  res->kernelBandwidth += (ke_acc_Bytes / ke_ms / 1e6) / mcin.Ns;
  res->gpuTime += (cu_ms + ke_ms) / mcin.Ns;
  res->gpuBandwidth += ((cuBytes + ke_acc_Bytes) / (cu_ms + ke_ms) / 1e6) / mcin.Ns;

  return reduced_lhood;  
}

// // tune rwsdf for a target acceptance ratio
void tune_ess_sp(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, gpu_v_str gpu, float *initCond, int length)
{
  // print_gpu_info();
  cudaSetDevice(0);

  cublasHandle_t handle;
  cublasCreate(&handle);

  mcmc_int_v mclocv;
  mcmc_int mcloc;
  mcloc.cposteriorf = 0;
  mcloc.pposteriorf = 0;
  mcloc.acceptancef = 0;
  mcloc.uf = 0;
  malloc_mcmc_vectors_sp(&mclocv, mcin);

  // set up the gpu vectors
  sz_str sz;
  dev_v_str d;

  getBlocksAndThreads(gpu.kernel, mcin.Nd, gpu.maxBlocks, gpu.maxThreads, &gpu.blocks, &gpu.threads);

  sz.samples = mcin.ddata * sizeof(float);
  sz.data = mcin.ddata * mcin.Nd * sizeof(float);
  sz.cuLhood = mcin.Nd * sizeof(float);
  sz.lhood = gpu.blocks * sizeof(float);

  cudaMalloc(&d.samplesf, sz.samples);
  cudaMalloc(&d.dataf, sz.data);
  cudaMalloc(&d.cuLhoodf, sz.cuLhood);    // kernel will return a vector of likelihoods  
  cudaMalloc(&d.lhoodf, sz.lhood);

  cudaMemcpy(d.dataf, data.dataf, sz.data, cudaMemcpyHostToDevice);  

  int chain_length = length;
  int runs = 40;
  float target_a[] = {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50};
  float error_tolerance = 0.01;
  float min_error = 9999999;
  float max_ess = -9999999;
  float lagidx = 500;

  float sd = mct->rwsdf;
  float ess_sd = sd;

  int accepted_samples, run, a_idx;
  float acc_ratio_c, acc_error_c, best_acc_ratio;
  float circ_sum, best_sd, ess_c;

  float *samples = NULL;
  samples = (float*) malloc(mcin.ddata * chain_length * sizeof(float));
  if(samples == NULL)
    fprintf(stderr, "ERROR: Samples vector did not allocated.\n");
  float *autocorr_lagk = NULL;
  autocorr_lagk = (float*) malloc(lagidx * sizeof(float));
  if(autocorr_lagk == NULL)
    fprintf(stderr, "ERROR: Autocorrelation vector did not allocated.\n");

  fprintf(stdout, "\nStarting tuning process. rwsdf = %5.3f\n", sd);
  
  for(a_idx=0; a_idx<9; a_idx++){
    fprintf(stdout, "\tStarting tuning for target ratio = %4.3f. Current rwsdf = %5.3f\n", target_a[a_idx], sd);    
    min_error = 9999999;
    for(run=0; run<runs; run++)
    {
      fprintf(stdout, "\t\tStarting Run %2d. Current rwsdf = %5.3f\n", run, sd);
      accepted_samples = 0;

      short_run_burn_in_sp(handle, r, mclocv, mcin, sd, &mcloc, sz, gpu, d, data.mvoutf, initCond);
      short_run_metropolis_sp(handle, r, mclocv, mcin, chain_length, sd, &mcloc, 
                                samples, &accepted_samples, sz, gpu, d, data.mvoutf);
      
      acc_ratio_c = accepted_samples/(float)chain_length;
      acc_error_c = fabs(acc_ratio_c - target_a[a_idx]);

      if(acc_error_c < min_error) // accept the current sd
      {
        best_sd = sd;
        min_error = acc_error_c;
        best_acc_ratio = acc_ratio_c;
        fprintf(stdout, "\t\t\tAccepted: rwsdf = %5.3f, acceptance = %4.3f, error = %4.3f\n", 
                        best_sd, best_acc_ratio, min_error);
      }else{
        fprintf(stdout, "\t\t\trwsdf = %5.3f, acceptance = %4.3f, error = %4.3f\n", 
                          sd, acc_ratio_c, acc_error_c);
      }
      
      if(min_error < error_tolerance) 
        break;
      
      sd *= acc_ratio_c/target_a[a_idx];
    }
    
    circ_sum = circular_autocorrelation_sp(autocorr_lagk, samples, mcin.ddata,
                                        chain_length, lagidx);
    ess_c = chain_length / (1 + 2*circ_sum);
    
    if(ess_c > max_ess)
    {
      max_ess = ess_c;
      ess_sd = sd;
      fprintf(stdout, "\tAccepted: ess = %8.3f, rwsdf = %5.3f\n", max_ess, ess_sd);
    }else{
      fprintf(stdout, "\tess= %8.3f, rwsdf = %5.3f\n", ess_c, sd);
    }
  }
  mct->rwsdf = ess_sd;
  fprintf(stdout, "Tuning finished. Selected rwsdf = %5.3f\n\n", mct->rwsdf);
  
  cublasDestroy(handle);
  free(samples);
  free(autocorr_lagk);
  cudaFree(d.samples);
  cudaFree(d.data);
  cudaFree(d.cuLhood);
  cudaFree(d.lhood);
  free_mcmc_vectors_sp(mclocv, mcin);
}


// // tune rwsdf for a target acceptance ratio
void tune_target_a_sp_v2(data_str data, gsl_rng *r, mcmc_str mcin, mcmc_tune_str *mct, 
                          gpu_v_str gpu, float *initCond, float ratio, int max_reps)
{
  // print_gpu_info();
  cudaSetDevice(0);

  cublasHandle_t handle;
  cublasCreate(&handle);

  mcmc_int_v mclocv;
  mcmc_int mcloc;
  mcloc.cposteriorf = 0;
  mcloc.pposteriorf = 0;
  mcloc.acceptancef = 0;
  mcloc.uf = 0;
  malloc_mcmc_vectors_sp(&mclocv, mcin);

  // set up the gpu vectors
  sz_str sz;
  dev_v_str d;

  getBlocksAndThreads(gpu.kernel, mcin.Nd, gpu.maxBlocks, gpu.maxThreads, &gpu.blocks, &gpu.threads);

  sz.samples = mcin.ddata * sizeof(float);
  sz.data = mcin.ddata * mcin.Nd * sizeof(float);
  sz.cuLhood = mcin.Nd * sizeof(float);
  sz.lhood = gpu.blocks * sizeof(float);

  cudaMalloc(&d.samplesf, sz.samples);
  cudaMalloc(&d.dataf, sz.data);
  cudaMalloc(&d.cuLhoodf, sz.cuLhood);    // kernel will return a vector of likelihoods  
  cudaMalloc(&d.lhoodf, sz.lhood);

  cudaMemcpy(d.dataf, data.dataf, sz.data, cudaMemcpyHostToDevice);

  int chain_length = 5000;
  int runs = max_reps;
  float target_a = ratio;
  float error_tolerance = 0.01;
  float min_error = 9999999;

  float sd = mct->rwsdf;
  float best_sd = sd;
  int accepted_samples, run;
  float acc_ratio_c, acc_error_c, best_acc_ratio;

  float *samples = NULL;
  samples = (float*) malloc(mcin.ddata * chain_length * sizeof(float));
  if(samples == NULL)
    fprintf(stderr, "ERROR: Samples vector did not allocated.\n");

  fprintf(stdout, "\nStarting tuning process. rwsdf = %5.3f\n", sd);
  
  for(run=0; run<runs; run++)
  {
    fprintf(stdout, "\tStarting Run %2d. Current rwsdf = %5.3f\n", run, sd);
    accepted_samples = 0;

    short_run_burn_in_sp(handle, r, mclocv, mcin, sd, &mcloc, sz, gpu, d, data.mvoutf, initCond);
    short_run_metropolis_sp(handle, r, mclocv, mcin, chain_length, sd, &mcloc, 
                              samples, &accepted_samples, sz, gpu, d, data.mvoutf);

    acc_ratio_c = accepted_samples/(float)chain_length;
    acc_error_c = fabs(acc_ratio_c - target_a);

    if(acc_error_c < min_error) // accept the current sd
    {
      best_sd = sd;
      min_error = acc_error_c;
      best_acc_ratio = acc_ratio_c;
      fprintf(stdout, "\t\tAccepted: rwsdf = %5.3f, acceptance = %4.3f, error = %4.3f\n", 
                      best_sd, best_acc_ratio, min_error);
    }else{
      fprintf(stdout, "\t\trwsdf = %5.3f, acceptance = %4.3f, error = %4.3f\n", 
                        sd, acc_ratio_c, acc_error_c);
    }
    
    if(min_error < error_tolerance) 
      break;
    
    sd *= acc_ratio_c/target_a;
  }

  mct->rwsdf = best_sd;
  fprintf(stdout, "Tuning finished. Selected rwsdf = %5.3f\n\n", mct->rwsdf);
  
  cublasDestroy(handle);
  free(samples);
  cudaFree(d.samples);
  cudaFree(d.data);
  cudaFree(d.cuLhood);
  cudaFree(d.lhood);
  free_mcmc_vectors_sp(mclocv, mcin);
}


void short_run_burn_in_sp(cublasHandle_t handle, gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, float sd, mcmc_int *mcloc, 
                            sz_str sz, gpu_v_str gpu, dev_v_str d, float *host_lhoodf, float *initCond)
{
  int i, dim_idx;
  float plhood, clhood;

  out_str res;

  // initialize burn in sequence
  for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
    mclocv.currentf[dim_idx] = 0;

  clhood = gpu_likelihood_f(handle, mcin, gpu, mclocv.currentf, sz.samples, d, host_lhoodf, &res); 
  // calculate the current posterior
  mcloc->cposteriorf = log_prior_sp(mclocv.currentf, mcin) + clhood;

  // start burn-in
  for(i=1; i<mcin.burnin; i++)
  {
    for(dim_idx = 0; dim_idx < mcin.ddata; dim_idx++){
      mclocv.proposedf[dim_idx] = mclocv.currentf[dim_idx] 
                                  + (float) gsl_ran_gaussian_ziggurat(r, (double)sd); // random walk using Marsaglia-Tsang ziggurat algorithm  
    }

    plhood = gpu_likelihood_f(handle, mcin, gpu, mclocv.proposedf, sz.samples, d, host_lhoodf, &res);

    mcloc->acceptancef = acceptance_ratio_sp(mclocv, mcloc, mcin, plhood);
    mcloc->uf = gsl_rng_uniform(r);

    if(log(mcloc->uf) <= mcloc->acceptancef)    // decide if you accept the proposed theta or not
    {
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++){
        mclocv.currentf[dim_idx] = mclocv.proposedf[dim_idx];
      }
      mcloc->cposteriorf = mcloc->pposteriorf; // make proposed posterior the current 
    }
  }
}


void short_run_metropolis_sp(cublasHandle_t handle, gsl_rng *r, mcmc_int_v mclocv, mcmc_str mcin, int chain_length, float sd, 
                              mcmc_int *mcloc, float *samples, int *accepted_samples, sz_str sz, 
                              gpu_v_str gpu, dev_v_str d, float *host_lhoodf)
{
  int i, dim_idx;
  float plhood;
  
  out_str res;

  // start metropolis
  for(i=0; i < chain_length; i++){    
    for(dim_idx = 0; dim_idx < mcin.ddata; dim_idx++){
      mclocv.proposedf[dim_idx] = mclocv.currentf[dim_idx] 
                                  + (float) gsl_ran_gaussian_ziggurat(r, (double)sd); // random walk using Marsaglia-Tsang ziggurat algorithm    
    }

    plhood = gpu_likelihood_f(handle, mcin, gpu, mclocv.proposedf, sz.samples, d, host_lhoodf, &res);

    mcloc->acceptancef = acceptance_ratio_sp(mclocv, mcloc, mcin, plhood);
    mcloc->uf = gsl_rng_uniform(r);

    if(log(mcloc->uf) <= mcloc->acceptancef)    // decide if you accept the proposed theta or not
    {
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++){
        mclocv.currentf[dim_idx] = mclocv.proposedf[dim_idx];
        samples[i*mcin.ddata + dim_idx] = mclocv.proposedf[dim_idx];
      }
      mcloc->cposteriorf = mcloc->pposteriorf; // make proposed posterior the current 
      *accepted_samples += 1; 
    }else{
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++){
        samples[i*mcin.ddata + dim_idx] = mclocv.currentf[dim_idx];
      }      
    }     
  }
}

float acceptance_ratio_sp(mcmc_int_v mclocv, mcmc_int *mcloc, mcmc_str mcin, float plhood) 
{
  float log_ratio;
  mcloc->pposteriorf = log_prior_sp(mclocv.proposedf, mcin) + plhood;
  log_ratio = mcloc->pposteriorf - mcloc->cposteriorf;

  return log_ratio;
}

float log_prior_sp(float *sample, mcmc_str mcin)
{ 
  float log_prob = 0;
  int dim_idx;

  for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++){  //assuming iid priors
    log_prob += log(gsl_ran_gaussian_pdf(sample[dim_idx], PRIOR_SD));
  }

  return log_prob;
}

#endif // __MCMC_GPU_SP_CU__