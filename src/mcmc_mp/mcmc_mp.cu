#ifndef __MCMC_MP_CU__
#define __MCMC_MP_CU__

#include "mcmc_mp.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

const int PRIOR_SD = 5;
const int MIN_BLOCKSZ = 64;
const int MAX_BLOCKSZ = 1024;

int MAX_THREADS = 256;
int MAX_BLOCKS = 64;
int CPU_THRESH = 32;
int LOWERBOUND_REPS = 50000;

__device__ double a_dev = 1.0;
__device__ double b_dev = 0.0;

__global__ void mcmc_mp_kernel(double *u, double *dataIn_d, double *samples_d, int8_t *z, 
															 int ResampFactor, int ResampChoose, int dataBlockSz, 
															 int dataBlocks, int dataDim, int datapoints, 
                               float *dotS, double *dotD, double *LBb, float *LDb, 
                               int *dataCounters, double e)
{
		__shared__ float sLDblock[1024];
		__shared__ double sLBblock[1024];
    __shared__ double sResamp[1024];
    __shared__ double sLB[1024];
	
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;
    int dotIdx = blockIdx.x * dataBlockSz;
    int matrixIdx = blockIdx.x * dataBlockSz;
    double ResampReg = 0;
    int cDark, cBright, cU;

    if(z[bidx]==0)  // dark point
    {
        if(bidx % ResampFactor == ResampChoose)    // resample dark point
        {
            // get the dot product of each datapoint inside the block
            if(tidx==0)
            {
                cublasHandle_t cpnHandle;
                cublasCreate(&cpnHandle);
                cublasDgemv(cpnHandle, CUBLAS_OP_N, dataBlockSz, dataDim, &a_dev, &dataIn_d[matrixIdx], datapoints, samples_d, 1, &b_dev, &dotD[dotIdx], 1);
                cudaDeviceSynchronize();
                cublasDestroy(cpnHandle);
            }
            __syncthreads();

            if(tidx<dataBlockSz) // block log-likelihood in double and single precision
            {
                sLDblock[tidx] = log((1 / (1 + exp(dotS[dotIdx + tidx]))) - e);
                sLBblock[tidx] = log(1 / (1 + exp(dotD[dotIdx + tidx])));
                sResamp[tidx] = log(sLDblock[tidx]) - log(sLBblock[tidx]);
            }else{
                sLDblock[tidx] = 0;
                sResamp[tidx] = 0;
            }

            __syncthreads();

            // perform block reduction both on sDataF and sDataD to accumulate block log-likelihoods
            for (unsigned int s=blockDim.x/2; s>0; s>>=1)
            {
                if (tidx < s)
                {
                    sResamp[tidx] += sResamp[tidx+s];
                    sLDblock[tidx] += sLDblock[tidx + s];
                }
                __syncthreads();
            }

            if (tidx == 0)
            {
                cDark = atomicAdd(&dataCounters[3], 1); // to ensure atomic access and unique index
                cU = atomicAdd(&dataCounters[4],1);      // increase number of u data accessed

                LDb[cDark] = sLDblock[0];   // store log-likelihood for dark block to perform reduction later

                ResampReg = 1 - exp(sResamp[0]);
                if(ResampReg <= u[cU])
                {  
                    z[bidx] = 0;
                    atomicAdd(&dataCounters[2], 1); // increase total dark number
                }else{
                    z[bidx] = 1;
                    atomicAdd(&dataCounters[0], 1); // increase total bright number
                }
            }
        }else{
            if(tidx<dataBlockSz)    // block likelihood in single precision
            {
                sLDblock[tidx] = log((1 / (1 + exp(dotS[dotIdx + tidx]))) - e);
            }else{
                sLDblock[tidx] = 0;
            }
            __syncthreads();

            // perform block reduction to accumulate single block log-likelihood
            for (unsigned int s=blockDim.x/2; s>0; s>>=1)
            {
                if (tidx < s)
                {
                    sLDblock[tidx] += sLDblock[tidx + s];
                }
                __syncthreads();
            }

            // write result for this block to global mem
            if (tidx == 0)
            {
                cDark = atomicAdd(&dataCounters[3], 1); // increase number of dark data accessed

                LDb[cDark] = sLDblock[0]; // return block log-likelihood for dark block to perform reduction later
                z[bidx] = 0;

                atomicAdd(&dataCounters[2], 1); // increase total dark data number
            }
        }
    }else{
        // get the dot product of each datapoint inside the block
        if(tidx==0)
        {
            cublasHandle_t cpnHandle;
            cublasCreate(&cpnHandle);
            cublasDgemv(cpnHandle, CUBLAS_OP_N, dataBlockSz, dataDim, &a_dev, &dataIn_d[matrixIdx], datapoints, samples_d, 1, &b_dev, &dotD[dotIdx], 1);
            cudaDeviceSynchronize();
            cublasDestroy(cpnHandle);
        }
        __syncthreads(); 
        // if(blockIdx.x==0)
        //     printf("tidx = %d, dot = %0.64f\n", tidx, dotD[dotIdx+tidx]);
        if(tidx<dataBlockSz)  // block log-likelihood in double and single precision
        {
            sLDblock[tidx] = (1 / (1 + exp(dotS[dotIdx + tidx]))) - e;
            sLBblock[tidx] = 1 / (1 + exp(dotD[dotIdx + tidx]));
            sResamp[tidx] = log(sLDblock[tidx]) - log(sLBblock[tidx]);
            sLB[tidx] = log(sLBblock[tidx] - sLDblock[tidx]);
            // if(blockIdx.x==0)
            //   printf("LB = %0.64f, LD = %0.64f, diff = %0.64f\n", sLBblock[tidx], sLDblock[tidx], sLBblock[tidx] - sLDblock[tidx]);
        }else{
            sLB[tidx] = 0;
            sResamp[tidx] = 0;
        }

        // printf("LB = %.64f, LD = %.64f\n", sLBblock[tidx], sLDblock[tidx]);
        __syncthreads();  // all threads have to be finished in order to perform reduction on the data
        // if(blockIdx.x == 0)
        //   printf("dotx+tidx = %d, tidx = %d, sLDblock = %.32f diff = %0.64f\n", dotIdx+tidx, tidx, sLDblock[tidx], sLBblock[tidx]-sLDblock[tidx]);
        unsigned int s;
        for (s=blockDim.x/2; s>0; s>>=1)  // accumulate current block log-likelihood
        {
            if (tidx < s)
            {
                sResamp[tidx] += sResamp[tidx + s];
                sLB[tidx] += sLB[tidx+s];
                // if(blockIdx.x==0)
                  // printf("LB = %0.64f, LD = %0.64f, logdiff = %0.64f\n", sLBblock[tidx+s], sLDblock[tidx+s], log(sLBblock[tidx+s] - sLDblock[tidx+s]));
            }
            __syncthreads();
        }

        // __syncthreads();
        // if(blockIdx.x == 0 && tidx ==0)
          // printf("sLB = %.64f, resamp = %0.64f\n", sLB[0], 1-exp(sResamp[0]));
        // write result for this block to global mem
        if (tidx == 0)
        {
            cBright = atomicAdd(&dataCounters[1], 1);  // increase number of bright data accessed
            cU = atomicAdd(&dataCounters[4],1);        // increase number of u data accessed
            
            LBb[cBright] = sLB[0];

            ResampReg = 1 - exp(sResamp[0]);
            if( ResampReg <= u[cU])
            {  
                z[bidx] = 0;
                atomicAdd(&dataCounters[2], 1);    // increase number of dark data
            }else{
                z[bidx] = 1;
                atomicAdd(&dataCounters[0], 1);             // increase number of bright data
            }
        }
        // __syncthreads();
        // if(blockIdx.x == 0)
        // printf("tidx = %d, LB = %0.64f\n", tidx, LBb[tidx]);
        // __syncthreads();
    }
}

__global__ void lowerbound_kernel_lvl1(double *dotD, float *dotS, int N,
                                  			double *LB, double *epsilon)
{
  __shared__ double sLBn[1024];
  __shared__ double sEpsilon[1024];

  int tidx = threadIdx.x;
  int didx = blockIdx.x * blockDim.x + threadIdx.x;

  if(didx < N)
  {
    sLBn[tidx] = 1/ (1 + exp(dotD[didx]));
    sEpsilon[tidx] = sLBn[tidx] - (1 / (1 + exp(dotS[didx])));
  }else{
    sLBn[tidx] = 0;
    sEpsilon[tidx] = 0;
  }
  __syncthreads();


  // perform block reduction both on sDataF and sDataD to accumulate block log-likelihoods
  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
  {
      if (tidx < s)
      {
          sLBn[tidx] += sLBn[tidx + s];

          if((sEpsilon[tidx] > sEpsilon[tidx+s]) && (sEpsilon[tidx] < 0))
            sEpsilon[tidx] = sEpsilon[tidx+s];
      }
      __syncthreads();
  }

  if(tidx == 0)
  {
    LB[blockIdx.x] = sLBn[0];
    epsilon[blockIdx.x] = sEpsilon[0];
  }
}

// performs reduction on likelihood vector and establishes max of epsilon
__global__ void lowerbound_kernel(double *LB, double *epsilon, int N, double *LBout, double *epsilonOut)
{
  __shared__ double sLBn[1024];
  __shared__ double sEpsilon[1024];
  
  int tidx = threadIdx.x;
  int didx = blockIdx.x * blockDim.x + threadIdx.x;

  if(didx < N)
  {
    sLBn[tidx] = LB[didx];
    sEpsilon[tidx] = epsilon[didx];
  }else{
    sLBn[tidx] = 0;
    sEpsilon[tidx] = 0;
  }
  __syncthreads();


  // perform block reduction both on sDataF and sDataD to accumulate block log-likelihoods
  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
  {
      if (tidx < s)
      {
          sLBn[tidx] += sLBn[tidx + s];

          if(sEpsilon[tidx] > sEpsilon[tidx+s] && sEpsilon[tidx]<0)
            sEpsilon[tidx] = sEpsilon[tidx+s];
      }
      __syncthreads();
  }

  if(tidx == 0)
  {
    LBout[blockIdx.x] = sLBn[0];
    epsilonOut[blockIdx.x] = sEpsilon[0];
  }
}


void mp_sampler(data_str data, gsl_rng *r, mp_str *mpVar, mcmc_str mcin,
                mcmc_tune_str mct, mcmc_v_str mcdata, double lhoodBound, out_str *res)
{
  cudaSetDevice(0);
  gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024));

  int accepted_samples;
  clock_t startBurn, stopBurn;
  clock_t startMcmc, stopMcmc;

  mcmc_int_v mclocv;
  mcmc_int mcloc;
  mcloc.cposterior = 0; mcloc.pposterior = 0; mcloc.acceptance = 0; mcloc.u = 0;
  malloc_mcmc_vectors(&mclocv, mcin);

  double *host_lhood = (double *) malloc(mpVar->dataBlockSz * sizeof(double));
  float *host_lhoodf = (float *) malloc(mpVar->dataBlockSz * sizeof(float));

  // set up the gpu vectors
  dev_v_str d;
  int *dataCounters;

  gpuErrchk(cudaMallocManaged(&dataCounters, 5 * sizeof(int)));  /* 0: totalBright, 2: total Dark, 4: how many u were accessed */
  gpuErrchk(cudaMalloc(&d.samples, mcin.ddata*sizeof(double)));           
  gpuErrchk(cudaMalloc(&d.samplesf, mcin.ddata*sizeof(float)));
  gpuErrchk(cudaMalloc(&d.data, mcin.ddata * mcin.Nd * sizeof(double)));                 
  gpuErrchk(cudaMalloc(&d.dataf, mcin.ddata * mcin.Nd * sizeof(float)));
  gpuErrchk(cudaMalloc(&d.dotS, mcin.Nd*sizeof(float)));
  gpuErrchk(cudaMalloc(&d.dotD, mcin.Nd*sizeof(double)));
  /* Allocate Nd doubles on device for u */
  gpuErrchk(cudaMalloc(&d.u, (mpVar->dataBlocks + ceil(mpVar->dataBlocks / mpVar->ResampFactor)) * sizeof(double)));
  gpuErrchk(cudaMalloc(&d.z, mpVar->dataBlocks * sizeof(int8_t)));
  gpuErrchk(cudaMalloc(&d.LDb, mpVar->dataBlocks * sizeof(float)));
  gpuErrchk(cudaMalloc(&d.LBb, mpVar->dataBlocks * sizeof(double)));
  gpuErrchk(cudaMalloc(&d.redLDb, mpVar->dataBlocks * sizeof(float)));
  gpuErrchk(cudaMalloc(&d.redLBb, mpVar->dataBlocks * sizeof(double)));

  gpuErrchk(cudaMemcpy(d.data, data.data, mcin.ddata * mcin.Nd * sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d.dataf, data.dataf, mcin.ddata * mcin.Nd * sizeof(float), cudaMemcpyHostToDevice));
  /* Set all z indexes to 1 indicating all blocks are bright*/
  gpuErrchk(cudaMemset(d.z, 1, mpVar->dataBlocks * sizeof(int8_t)));

  curandGenerator_t gen;
  /* Create pseudo-random number generator */
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  /* Set seed */
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

  /* Initialize number of bright and dark blocks. 
   * At the beginning all blocks should be bright */
  mpVar->brightBlocks = mpVar->dataBlocks;
  mpVar->darkBlocks = 0;

  startBurn = clock();
  if(mcin.burnin != 0)
    burn_in_metropolis_mp(gen, r, mpVar, mcin, mct, mcdata, mclocv, &mcloc, 
    						d, host_lhood, host_lhoodf, dataCounters, lhoodBound);
  stopBurn = clock() - startBurn;

  accepted_samples = 0;  

  startMcmc = clock();
  metropolis_mp(gen, r, mpVar, mcin, mct, mcdata, mclocv, &mcloc, &accepted_samples, 
  								d, host_lhood, host_lhoodf, dataCounters, lhoodBound, res);
  stopMcmc = clock() - startMcmc;

  res->burnTime = stopBurn * 1000 / CLOCKS_PER_SEC;   // burn in time in ms
  res->mcmcTime = stopMcmc * 1000 / CLOCKS_PER_SEC;   // mcmc time in ms
  res->acceptance = (double)accepted_samples / mcin.Ns;
  
  gpuErrchk(cudaFree(d.samples));			gpuErrchk(cudaFree(d.samplesf));
  gpuErrchk(cudaFree(d.data));              gpuErrchk(cudaFree(d.dataf));
  gpuErrchk(cudaFree(d.dotS));				gpuErrchk(cudaFree(d.dotD));
  gpuErrchk(cudaFree(d.LDb));				gpuErrchk(cudaFree(d.LBb));
  gpuErrchk(cudaFree(d.redLDb));			gpuErrchk(cudaFree(d.redLBb));
  gpuErrchk(cudaFree(d.z));
  gpuErrchk(cudaFree(d.u));
  gpuErrchk(cudaFree(dataCounters));

  curandDestroyGenerator(gen);
	free(host_lhood);
	free(host_lhoodf);  
  free_mcmc_vectors(mclocv, mcin);
}

void metropolis_mp(curandGenerator_t gen, gsl_rng *r, mp_str *mpVar, mcmc_str mcin, mcmc_tune_str mct, 
					mcmc_v_str mcdata, mcmc_int_v mclocv, mcmc_int *mcloc, int *accepted_samples,
					dev_v_str d, double *host_lhood, float *host_lhoodf, int *dataCounters, double lhoodBound, 
					out_str *res)
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
    {
      // random walk using Marsaglia-Tsang ziggurat algorithm
      mclocv.proposed[dim_idx] = mclocv.current[dim_idx] + gsl_ran_gaussian_ziggurat(r, mct.rwsd);
      mclocv.proposedf[dim_idx] = mclocv.proposed[dim_idx];
    }

    plhood = mp_likelihood(gen, r, mpVar, mcin, mclocv.proposed, mcin.ddata*sizeof(double), 
                           mclocv.proposedf, mcin.ddata*sizeof(float), d, host_lhood, host_lhoodf, 
                           dataCounters, lhoodBound, res);
    
    // calculate acceptance ratio
    mcloc->acceptance = acceptance_ratio_mp(mclocv, mcloc, mcin, plhood);
    
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

void burn_in_metropolis_mp(curandGenerator_t gen, gsl_rng *r, mp_str *mpVar, mcmc_str mcin, mcmc_tune_str mct, 
							mcmc_v_str mcdata, mcmc_int_v mclocv, mcmc_int *mcloc, dev_v_str d, 
							double *host_lhood, float *host_lhoodf, int *dataCounters, double lhoodBound)
{
  int i, dim_idx;
  double plhood, clhood;
  out_str res;

  fprintf(stdout, "Starting burn in process. Selected rwsd = %f\n", mct.rwsd);
  // initialize burn in sequence
  for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
  {
    // mclocv.current[dim_idx] = mcdata.burn[dim_idx];
    mclocv.current[dim_idx] = 1;
    mclocv.currentf[dim_idx] = mclocv.current[dim_idx];
  }

  clhood = mp_likelihood(gen, r, mpVar, mcin, mclocv.current, mcin.ddata*sizeof(double), 
                         mclocv.currentf, mcin.ddata*sizeof(float), d, host_lhood, 
                         host_lhoodf, dataCounters, lhoodBound, &res);
  // calculate the current posterior
  mcloc->cposterior = log_prior_mp(mclocv.current, mcin) + clhood;

  // start burn in
  for(i=1; i<mcin.burnin; i++)
  {
    // propose next sample
    for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++){
      mclocv.proposed[dim_idx] = mclocv.current[dim_idx] + gsl_ran_gaussian_ziggurat(r, mct.rwsd); // random walk using Marsaglia-Tsang ziggurat algorithm
      mclocv.proposedf[dim_idx] = mclocv.proposed[dim_idx];
    }

    plhood = mp_likelihood(gen, r, mpVar, mcin, mclocv.proposed, mcin.ddata*sizeof(double), 
                           mclocv.proposedf, mcin.ddata*sizeof(float), d, host_lhood, 
                           host_lhoodf, dataCounters, lhoodBound, &res);

    mcloc->acceptance = acceptance_ratio_mp(mclocv, mcloc, mcin, plhood);
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

double mp_likelihood(curandGenerator_t gen, gsl_rng *r, mp_str *mpVar, mcmc_str mcin, 
                     double *samples, size_t sampleSz, float *samplesf, size_t sampleSzf, 
                     dev_v_str d, double *host_lhood, float *host_lhoodf,
                     int *dataCounters, double lhoodBound, out_str *res)
{
  double red_d = 0.0;
  float red_f = 0.0;
  float a = 1.0;
  float b = 0.0;
  double mp_lhood = 0;
	double ke_acc_Bytes = 0;

	gpu_v_str gpu;
	gpu.maxBlocks = MAX_BLOCKS;
	gpu.maxThreads = MAX_THREADS;
	gpu.kernel = SequentialReduction;
  gpu.cpuThresh = CPU_THRESH;

  int uResampBlocks = ceil(mpVar->dataBlocks / mpVar->ResampFactor);
  int numUnif = mpVar->brightBlocks + uResampBlocks;  // # of random numbers to generate

  dataCounters[0] = 0; // reset counter of bright data  
  dataCounters[1] = 0;  
  dataCounters[2] = 0; // reset counter of dark data
  dataCounters[3] = 0;
  dataCounters[4] = 0; // reset the number of u data accessed

	/* Chooses which blocks to resample.
	 * Ensures all blocks will be resample eventually */
  int ResampChoose = (int)gsl_rng_uniform_int(r, mpVar->ResampFactor);

  /* Generate doubles on device */
  curandGenerateUniformDouble(gen, d.u, numUnif);

  gpuErrchk(cudaMemcpy(d.samples, samples, mcin.ddata*sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d.samplesf, samplesf, mcin.ddata*sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  cublasCreate(&handle);
  // get dot product of all datapoints
  cublasSgemv(handle, CUBLAS_OP_N, mcin.Nd, mcin.ddata, &a, d.dataf, mcin.Nd, d.samplesf, 1, &b, d.dotS, 1);
  gpuErrchk(cudaDeviceSynchronize());
  cublasDestroy(handle);

  gpu.blocks = mpVar->dataBlocks;
  gpu.threads = next2(mpVar->dataBlockSz);
  // printf("%d\n", gpu.threads);

  mcmc_mp_kernel<<< gpu.blocks, gpu.threads >>>(d.u, d.data, d.samples, d.z, mpVar->ResampFactor, 
                                          			ResampChoose, mpVar->dataBlockSz, mpVar->dataBlocks, 
                                          			mcin.ddata, mcin.Nd, d.dotS, d.dotD, d.LBb, d.LDb, 
                                          			dataCounters, lhoodBound);
  gpuErrchk(cudaDeviceSynchronize());
  printf("kernel done!!!!!!!!!!!!!!!!!!\n");
  // perform reduction to d.LBb and d.LDb to get single lhood
  if(mpVar->brightBlocks != 0)
  {
    gpu.size = mpVar->brightBlocks;
    getBlocksAndThreads(gpu.kernel, gpu.size, gpu.maxBlocks, gpu.maxThreads, &gpu.blocks, &gpu.threads);
    red_d = reduction_mp_d(gpu, d, host_lhood, &ke_acc_Bytes);
  }

  printf("Reduction level 1 Done!!!!!!!!!!!!!!!!!!!!!!!!!\n");

  if(mpVar->darkBlocks != 0)
  {  
    gpu.size = mpVar->darkBlocks;
    getBlocksAndThreads(gpu.kernel, gpu.size, gpu.maxBlocks, gpu.maxThreads, &gpu.blocks, &gpu.threads);
    red_f = reduction_mp_f(gpu, d, host_lhoodf, &ke_acc_Bytes);
  }

  mp_lhood = red_d + red_f;
  mpVar->brightBlocks = dataCounters[0];
  mpVar->darkBlocks = dataCounters[2];
  mpVar->ratio = ((float)(mpVar->brightBlocks))/(mpVar->dataBlocks);
  printf("Lhood = %f, blocks = %d, bright = %d, dark = %d\n", mp_lhood, mpVar->dataBlocks, mpVar->brightBlocks, mpVar->darkBlocks);
  return mp_lhood;  
}

double reduction_mp_d(gpu_v_str gpu, dev_v_str d, double *host_lhood, double *ke_acc_Bytes)
{
  double gpu_result = 0;
  int i;
  int numBlocks = gpu.blocks;

  *ke_acc_Bytes = gpu.size * sizeof(double);

  reduceSum_d(gpu.size, gpu.threads, gpu.blocks, gpu.kernel, d.LBb, d.redLBb, Reduction);  // perform log-reduction on first step for bright points
 
  while(numBlocks >= gpu.cpuThresh)
  {
    getBlocksAndThreads(gpu.kernel, numBlocks, gpu.maxBlocks, gpu.maxThreads, &gpu.blocks, &gpu.threads);
    
    ke_acc_Bytes += numBlocks * sizeof(double);
    
    gpuErrchk(cudaMemcpy(d.LBb, d.redLBb, numBlocks*sizeof(double), cudaMemcpyDeviceToDevice));
    reduceSum_d(numBlocks, gpu.threads, gpu.blocks, gpu.kernel, d.LBb, d.redLBb, Reduction);    
    if(gpu.kernel < 3)
    {
      numBlocks = (numBlocks + gpu.threads - 1) / gpu.threads;
    }else{
      numBlocks = (numBlocks +(gpu.threads*2-1)) / (gpu.threads*2);
    }
  }

  gpuErrchk(cudaMemcpy(host_lhood, d.redLBb, numBlocks*sizeof(double), cudaMemcpyDeviceToHost));  

  // accumulate result on CPU
  for(i=0; i<numBlocks; i++){
    gpu_result += host_lhood[i];
  }

  return gpu_result;
}

float reduction_mp_f(gpu_v_str gpu, dev_v_str d, float *host_lhoodf, double *ke_acc_Bytes)
{
  float gpu_result = 0;
  int i;
  int numBlocks = gpu.blocks;

  *ke_acc_Bytes = gpu.size * sizeof(float);

  reduceSum_f(gpu.size, gpu.threads, gpu.blocks, gpu.kernel, d.LDb, d.redLDb, Reduction);

  while(numBlocks >= gpu.cpuThresh)
  {
    getBlocksAndThreads(gpu.kernel, numBlocks, gpu.maxBlocks, gpu.maxThreads, &gpu.blocks, &gpu.threads);
    
    ke_acc_Bytes += numBlocks * sizeof(float);
    
    gpuErrchk(cudaMemcpy(d.LDb, d.redLDb, numBlocks*sizeof(float), cudaMemcpyDeviceToDevice));

    reduceSum_f(numBlocks, gpu.threads, gpu.blocks, gpu.kernel, d.LDb, d.redLDb, Reduction);    

    if(gpu.kernel < 3)
    {
      numBlocks = (numBlocks + gpu.threads - 1) / gpu.threads;
    }else{
      numBlocks = (numBlocks +(gpu.threads*2-1)) / (gpu.threads*2);
    }

  }

  gpuErrchk(cudaMemcpy(host_lhoodf, d.redLDb, numBlocks*sizeof(float), cudaMemcpyDeviceToHost));  

  // accumulate result on CPU
  for(i=0; i<numBlocks; i++){
    gpu_result += host_lhoodf[i];
  }

  return gpu_result;
}

double acceptance_ratio_mp(mcmc_int_v mclocv, mcmc_int *mcloc, mcmc_str mcin, double plhood) 
{
  double log_ratio;
  mcloc->pposterior = log_prior_mp(mclocv.proposed, mcin) + plhood;
  log_ratio = mcloc->pposterior - mcloc->cposterior;

  return exp(log_ratio);
}

double log_prior_mp(double *sample, mcmc_str mcin)
{ 
  double log_prob = 0;
  int dim_idx;

  for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++){  //assuming iid priors
    log_prob += log(gsl_ran_gaussian_pdf(sample[dim_idx], PRIOR_SD));
  }

  return log_prob;
}


// returns the lower bound
double get_lowerbound(data_str data, double *initialCond, gsl_rng *r,mcmc_str mcin, mcmc_tune_str mct)
{
  cudaSetDevice(0);
  int i, dim_idx;
  double plhood, clhood;

  double global_epsilon = 0;

  mcmc_int_v mclocv;
  mcmc_int mcloc;
  mcloc.cposterior = 0; mcloc.pposterior = 0; mcloc.acceptance = 0; mcloc.u = 0;
  malloc_mcmc_vectors(&mclocv, mcin);

  double *epsilon = (double *) malloc(mcin.Nd * sizeof(double));
  double *host_lhood = (double *) malloc(mcin.Nd * sizeof(double));
  // set up the gpu vectors
  dev_v_str d;
  gpuErrchk(cudaMalloc(&d.samples, mcin.ddata*sizeof(double)));           
  gpuErrchk(cudaMalloc(&d.samplesf, mcin.ddata*sizeof(float)));
  gpuErrchk(cudaMalloc(&d.data, mcin.ddata * mcin.Nd * sizeof(double)));                 
  gpuErrchk(cudaMalloc(&d.dataf, mcin.ddata * mcin.Nd * sizeof(float)));
  gpuErrchk(cudaMalloc(&d.dotS, mcin.Nd*sizeof(float)));
  gpuErrchk(cudaMalloc(&d.dotD, mcin.Nd*sizeof(double)));
  gpuErrchk(cudaMalloc(&d.LB, mcin.Nd * sizeof(double)));
  gpuErrchk(cudaMalloc(&d.epsilon, mcin.Nd * sizeof(double)));
  gpuErrchk(cudaMalloc(&d.LBtemp, mcin.Nd * sizeof(double)));
  gpuErrchk(cudaMalloc(&d.epsilonTemp, mcin.Nd * sizeof(double)));

  gpuErrchk(cudaMemcpy(d.data, data.data, mcin.ddata * mcin.Nd * sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d.dataf, data.dataf, mcin.ddata * mcin.Nd * sizeof(float), cudaMemcpyHostToDevice));
  fprintf(stdout, "Starting lowerbound process. Current Lowerbound = %f\n", global_epsilon);
  
  // initialize burn in sequence
  for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
  {
    // mclocv.current[dim_idx] = initialCond[dim_idx];
    mclocv.current[dim_idx] = 0;
    mclocv.currentf[dim_idx] = mclocv.current[dim_idx];
  }

  clhood = likelihood_bound(r, mcin, mclocv.current, mcin.ddata*sizeof(double), mclocv.currentf, 
  													mcin.ddata*sizeof(float), d, host_lhood, epsilon);
  // calculate the current posterior
  mcloc.cposterior = log_prior_mp(mclocv.current, mcin) + clhood;

  global_epsilon = epsilon[0];
  printf("New Lowerbound = %.64f\n", global_epsilon);

  // start burn in
  for(i=1; i<LOWERBOUND_REPS; i++)
  {
    // propose next sample
    for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++){
      mclocv.proposed[dim_idx] = mclocv.current[dim_idx] + gsl_ran_gaussian_ziggurat(r, mct.rwsd); // random walk using Marsaglia-Tsang ziggurat algorithm
      mclocv.proposedf[dim_idx] = mclocv.proposed[dim_idx];
    }

    plhood = likelihood_bound(r, mcin, mclocv.proposed, mcin.ddata*sizeof(double), mclocv.proposedf, 
    													mcin.ddata*sizeof(float), d, host_lhood, epsilon);

    if(global_epsilon < epsilon[0])
    {
      global_epsilon = epsilon[0];
      printf("New Lowerbound = %.64f\n", global_epsilon); 
    }

    mcloc.acceptance = acceptance_ratio_mp(mclocv, &mcloc, mcin, plhood);
    mcloc.u = gsl_rng_uniform(r);
 
    if(mcloc.u <= mcloc.acceptance)
    {
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
      {
        mclocv.current[dim_idx] = mclocv.proposed[dim_idx];
      }
      mcloc.cposterior = mcloc.pposterior;
    }
  }

  fprintf(stdout, "Lowerbound process finished. Selected Lowerbound = %.64f\n", global_epsilon);

	gpuErrchk(cudaFree(d.samples));				gpuErrchk(cudaFree(d.samplesf));
	gpuErrchk(cudaFree(d.data));					gpuErrchk(cudaFree(d.dataf));
	gpuErrchk(cudaFree(d.dotS));					gpuErrchk(cudaFree(d.dotD));
  gpuErrchk(cudaFree(d.LB));						gpuErrchk(cudaFree(d.LBtemp));
  gpuErrchk(cudaFree(d.epsilon));			  gpuErrchk(cudaFree(d.epsilonTemp));
  
  free(epsilon);
  free(host_lhood);
  free_mcmc_vectors(mclocv, mcin);

  return global_epsilon;
}

double likelihood_bound(gsl_rng *r, mcmc_str mcin, double *samples_d, size_t sampleSz, 
												float *samples_f, size_t sampleSzf, dev_v_str d, 
												double *host_lhood, double *epsilon)
{
  double a = 1.0;
  double b = 0.0; 
  float af = 1.0;
  float bf = 0.0;
  double gpu_result = 0;

  gpuErrchk(cudaMemcpy(d.samples, samples_d, sampleSz, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d.samplesf, samples_f, sampleSzf, cudaMemcpyHostToDevice));
  
  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasDgemv(handle, CUBLAS_OP_N, mcin.Nd , mcin.ddata, &a, d.data, mcin.Nd, d.samples, 1, &b, d.dotD, 1);
  cublasSgemv(handle, CUBLAS_OP_N, mcin.Nd , mcin.ddata, &af, d.dataf, mcin.Nd, d.samplesf, 1, &bf, d.dotS, 1);            

  cublasDestroy(handle);

  gpu_result = call_boundKernel(mcin, d, host_lhood, epsilon);

  return gpu_result;
}

double call_boundKernel(mcmc_str mcin, dev_v_str d, double *host_lhood, double *epsilon)
{
  double gpu_result = 0;
  double e;
  int numBlocks;
  gpu_v_str gpu;
  gpu.size = mcin.Nd;
  gpu.maxBlocks = MAX_BLOCKS;
  gpu.maxThreads = MAX_THREADS;
  gpu.cpuThresh = CPU_THRESH;

  getBlocksAndThreads(SequentialReduction, gpu.size, gpu.maxBlocks, gpu.maxThreads, &gpu.blocks, &gpu.threads);  

  lowerbound_kernel_lvl1 <<< gpu.blocks, gpu.threads >>> (d.dotD, d.dotS, mcin.Nd, d.LB, d.epsilon);
  gpuErrchk(cudaDeviceSynchronize());

  numBlocks = gpu.blocks;

  while(numBlocks >= gpu.cpuThresh)
  {
    getBlocksAndThreads(SequentialReduction, numBlocks, gpu.maxBlocks, gpu.maxThreads, &gpu.blocks, &gpu.threads);

  	lowerbound_kernel <<< gpu.blocks, gpu.threads >>> (d.LB, d.epsilon, numBlocks, d.LBtemp, d.epsilonTemp);  

  	gpuErrchk(cudaMemcpy(d.LB, d.LBtemp, numBlocks * sizeof(double), cudaMemcpyDeviceToDevice));
  	gpuErrchk(cudaMemcpy(d.epsilon, d.epsilonTemp, numBlocks * sizeof(double), cudaMemcpyDeviceToDevice));

    if(gpu.kernel < 3)
    {
      numBlocks = (numBlocks + gpu.threads - 1) / gpu.threads;
    }else{
      numBlocks = (numBlocks +(gpu.threads*2-1)) / (gpu.threads*2);
    }
  }

  cudaMemcpy(host_lhood, d.LB, numBlocks * sizeof(double), cudaMemcpyDeviceToHost); 
  cudaMemcpy(epsilon, d.epsilon, numBlocks * sizeof(double), cudaMemcpyDeviceToHost); 
 
  e = epsilon[0];
  // accumulate result on CPU
  int i;
  for(i=0; i<numBlocks; i++){
    gpu_result += host_lhood[i];
    if(e>epsilon[i] && epsilon<0)
      e = epsilon[i];
  }
  epsilon[0] = e;

  return gpu_result;
}


void get_blockSz(data_str data, gsl_rng *r, mp_str *mpVar, mcmc_str mcin, 
								 mcmc_tune_str mct, double *initialCond, double lhoodBound)
{
  cudaSetDevice(0);
  gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024));

  mcmc_int_v mclocv;
  mcmc_int mcloc;
  mcloc.cposterior = 0; mcloc.pposterior = 0; mcloc.acceptance = 0; mcloc.u = 0;
  malloc_mcmc_vectors(&mclocv, mcin);

	mp_str mpVarLoc;

	clock_t startTune, stopTune;
	bool first_it = true;
	float time, bestTime;
	int bestBlockSz = 0;
	int bestResamp = 0;
	int resamp;

  double *host_lhood;
  host_lhood = (double *) malloc(mcin.Nd * sizeof(double));
  float *host_lhoodf = (float *) malloc(mcin.Nd * sizeof(float));

  // set up the gpu vectors
  dev_v_str d;
  int *dataCounters;

  gpuErrchk(cudaMallocManaged(&dataCounters, 5 * sizeof(int)));  /* 0: totalBright, 2: total Dark, 4: how many u were accessed */
  gpuErrchk(cudaMalloc(&d.samples, mcin.ddata*sizeof(double)));           
  gpuErrchk(cudaMalloc(&d.samplesf, mcin.ddata*sizeof(float)));
  gpuErrchk(cudaMalloc(&d.data, mcin.ddata * mcin.Nd * sizeof(double)));                 
  gpuErrchk(cudaMalloc(&d.dataf, mcin.ddata * mcin.Nd * sizeof(float)));
  gpuErrchk(cudaMalloc(&d.dotS, mcin.Nd*sizeof(float)));
  gpuErrchk(cudaMalloc(&d.dotD, mcin.Nd*sizeof(double)));

  gpuErrchk(cudaMemcpy(d.data, data.data, mcin.ddata * mcin.Nd * sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d.dataf, data.dataf, mcin.ddata * mcin.Nd * sizeof(float), cudaMemcpyHostToDevice));

  curandGenerator_t gen;
  /* Create pseudo-random number generator */
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  /* Set seed */
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

	for(mpVarLoc.dataBlockSz = MIN_BLOCKSZ; mpVarLoc.dataBlockSz <= MAX_BLOCKSZ; mpVarLoc.dataBlockSz *= 2)
	{
		mpVarLoc.dataBlocks = ceil(mcin.Nd/mpVarLoc.dataBlockSz);
		printf("mpVarLoc.dataBlocks = %d \n",mpVarLoc.dataBlocks);
	  gpuErrchk(cudaMalloc(&d.z, mpVarLoc.dataBlocks * sizeof(int8_t)));
	  gpuErrchk(cudaMalloc(&d.LDb, mpVarLoc.dataBlocks * sizeof(float)));
	  gpuErrchk(cudaMalloc(&d.LBb, mpVarLoc.dataBlocks * sizeof(double)));
	  gpuErrchk(cudaMalloc(&d.redLDb, mpVarLoc.dataBlocks * sizeof(float)));
	  gpuErrchk(cudaMalloc(&d.redLBb, mpVarLoc.dataBlocks * sizeof(double)));

	  for(resamp=mpVarLoc.dataBlocks/10; resamp>=10; resamp/=10)
		{
			mpVarLoc.ResampFactor = resamp;
		  /* Set all z indexes to 1 indicating all blocks are bright*/
		  gpuErrchk(cudaMemset(d.z, 1, mpVarLoc.dataBlocks * sizeof(int8_t)));
		  /* Allocate doubles on device for u */
		  gpuErrchk(cudaMalloc(&d.u, (mpVarLoc.dataBlocks + ceil(mpVarLoc.dataBlocks / mpVarLoc.ResampFactor)) * sizeof(double)));

		  mpVarLoc.brightBlocks = mpVarLoc.dataBlocks;
  		mpVarLoc.darkBlocks = 0;

  		startTune = clock();
		  short_run_mp(gen, r, &mpVarLoc, mcin, mct, initialCond, mclocv, &mcloc, d, host_lhood, host_lhoodf, dataCounters, lhoodBound);
  		stopTune = clock() - startTune;

			time = stopTune * 1000 / CLOCKS_PER_SEC;

			if(first_it)
			{
				bestTime = time;
				bestBlockSz = mpVarLoc.dataBlockSz;
				bestResamp = mpVarLoc.ResampFactor;
				first_it = false;
				printf("Update: New BlockSz = %d, New Resample Factor = %d\n", bestBlockSz, bestResamp);	
			}else{
				if(time<bestTime)
				{
					bestTime = time;
					bestBlockSz = mpVarLoc.dataBlockSz;
					bestResamp = mpVarLoc.ResampFactor;
					printf("Update: New BlockSz = %d, New Resample Factor = %d\n", bestBlockSz, bestResamp);		
				}else{
					printf("Current BlockSz = %d, Current Resample Factor = %d, Current Time = %fms, Best BlockSz = %d, Best Resample Factor = %d, Best Time = %fms\n",
									mpVarLoc.dataBlockSz, mpVarLoc.ResampFactor, time, bestBlockSz, bestResamp, bestTime);
				}			
			}
  	  gpuErrchk(cudaFree(d.u));
		}

	  gpuErrchk(cudaFree(d.z));
	  gpuErrchk(cudaFree(d.LDb));
	  gpuErrchk(cudaFree(d.LBb));
	  gpuErrchk(cudaFree(d.redLDb));
	  gpuErrchk(cudaFree(d.redLBb));
	}

	mpVar->dataBlockSz = bestBlockSz;
	mpVar->ResampFactor = bestResamp;
	mpVar->dataBlocks = ceil(mcin.Nd/bestBlockSz);

	printf("Block size and Resample Factor tuning finished. Selected Parameters: BlockSz = %d, DataBlocks = %d, Resample Factor = %d\n", 			
																																													bestBlockSz, mpVar->dataBlocks, bestResamp);

	gpuErrchk(cudaFree(d.samples));
	gpuErrchk(cudaFree(d.samplesf));
	gpuErrchk(cudaFree(d.data));
	gpuErrchk(cudaFree(d.dataf));
	gpuErrchk(cudaFree(d.dotS));
	gpuErrchk(cudaFree(d.dotD));

	free(dataCounters);
	free(host_lhood);
	free(host_lhoodf);
  free_mcmc_vectors(mclocv, mcin);
}

void short_run_mp(curandGenerator_t gen, gsl_rng *r, mp_str *mpVar, mcmc_str mcin, mcmc_tune_str mct, 
									double *initialCond, mcmc_int_v mclocv, mcmc_int *mcloc, dev_v_str d,
                  double *host_lhood, float *host_lhoodf, int *dataCounters, double lhoodBound)
{
  int i, dim_idx;
  int shortRunIdx = 500;
  double plhood, clhood;
  out_str res;

  fprintf(stdout, "Starting short run. Selected rwsd = %f\n", mct.rwsd);
  // initialize burn in sequence
  for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
  {
    // mclocv.current[dim_idx] = initialCond[dim_idx];
    mclocv.current[dim_idx] = 1;
    mclocv.currentf[dim_idx] = mclocv.current[dim_idx];
  }

  clhood = mp_likelihood(gen, r, mpVar, mcin, mclocv.current, mcin.ddata*sizeof(double), mclocv.currentf, 
  												mcin.ddata*sizeof(float), d, host_lhood, host_lhoodf, dataCounters, lhoodBound, &res);
      printf("plhood = %.64f\n", clhood);
  // calculate the current posterior
  mcloc->cposterior = log_prior_mp(mclocv.current, mcin) + clhood;

  // start burn in
  for(i=1; i<shortRunIdx; i++)
  {
  	printf("run = %d\n", i);
    // propose next sample
    for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++){
      mclocv.proposed[dim_idx] = mclocv.current[dim_idx] + gsl_ran_gaussian_ziggurat(r, mct.rwsd); // random walk using Marsaglia-Tsang ziggurat algorithm
      mclocv.proposedf[dim_idx] = mclocv.proposed[dim_idx];
    }

    plhood = mp_likelihood(gen, r, mpVar, mcin, mclocv.proposed, mcin.ddata*sizeof(double), mclocv.proposedf, 
    												mcin.ddata*sizeof(float), d, host_lhood, host_lhoodf, dataCounters, lhoodBound, &res);
    printf("plhood = %.64f\n", plhood);
    mcloc->acceptance = acceptance_ratio_mp(mclocv, mcloc, mcin, plhood);
    mcloc->u = gsl_rng_uniform(r);
 
    if(mcloc->u <= mcloc->acceptance)
    {
      for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++)
      {
        mclocv.current[dim_idx] = mclocv.proposed[dim_idx];
      }
      mcloc->cposterior = mcloc->pposterior;
    }
  }
  fprintf(stdout, "Short Run Completed.\n\n");
}
#endif // __MCMC_MP_CU__