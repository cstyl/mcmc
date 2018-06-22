#ifndef __MP_KERNEL_CU__
#define __MP_KERNEL_CU__

__device__ double a_dev = 1.0;
__device__ double b_dev = 0.0;

__global__ void mcmc_mp_kernel(double *u, double *dataIn_d, double *samples_d,
                               int8_t *z, int ResampFactor, int ResampChoose,
                               int dataBlockSz, int dataBlocks, int dataDim, int datapoints, 
                               float *dotS, double *dotD, double *LBb, float *LDb, int *dataCounters,
                               double e)
{
    // extern __shared__ float sarray[];
    // float* sLDblock = (float*) sarray;
    // // double needs to be aligned to 8 bytes
    // double* sLBblock = (blockDim.x<=32) ? (double*) &sarray[2*blockDim.x] : (double*) &sarray[blockDim.x];
	__shared__ float sLDblock[3072];
	__shared__ double sLBblock[3072];
	
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
                sLDblock[tidx] = logf((1 / (1 + expf(dotS[dotIdx + tidx]))) - e);
                sLBblock[tidx] = -log(1 + exp(dotD[dotIdx + tidx]));
            }else{
                sLDblock[tidx] = 0;
                sLBblock[tidx] = 0;
            }
            __syncthreads();

            // perform block reduction both on sDataF and sDataD to accumulate block log-likelihoods
            for (unsigned int s=blockDim.x/2; s>0; s>>=1)
            {
                if (tidx < s)
                {
                    sLDblock[tidx] += sLDblock[tidx + s];
                    sLBblock[tidx] += sLBblock[tidx + s];
                }
                __syncthreads();
            }

            // write result for this block to global mem
            if (tidx == 0)
            {
                cDark = atomicAdd(&dataCounters[3], 1); // to ensure atomic access and unique index
                cU = atomicAdd(&dataCounters[4],1);      // increase number of u data accessed

                LDb[cDark] = sLDblock[0];   // store log-likelihood for dark block to perform reduction later

                ResampReg = 1 - exp(sLDblock[0])/exp(sLBblock[0]);
                if(ResampReg <= u[cU])
                {  
                    z[bidx] = 1;
                    atomicAdd(&dataCounters[0], 1); // increase total bright number
                }else{
                    z[bidx] = 0;
                    atomicAdd(&dataCounters[2], 1); // increase total dark number
                }
            }
        }else{
            if(tidx<dataBlockSz)    // block likelihood in single precision
            {
                sLDblock[tidx] = logf((1 / (1 + expf(dotS[dotIdx + tidx]))) - e);
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

        if(tidx<dataBlockSz)  // block log-likelihood in double and single precision
        {
            sLDblock[tidx] = logf((1 / (1 + expf(dotS[dotIdx + tidx]))) - e);
            sLBblock[tidx] = -log(1 + exp(dotD[dotIdx + tidx]));
        }else{
            sLDblock[tidx] = 0;
            sLBblock[tidx] = 0;
        }
        __syncthreads();  // all threads have to be finished in order to perform reduction on the data

        unsigned int s;
        for (s=blockDim.x/2; s>0; s>>=1)  // accumulate current block log-likelihood
        {
            if (tidx < s)
            {
                sLDblock[tidx] += sLDblock[tidx + s];
                sLBblock[tidx] += sLBblock[tidx + s];
            }
            __syncthreads();
        }

        // write result for this block to global mem
        if (tidx == 0)
        {
            cBright = atomicAdd(&dataCounters[1], 1);  // increase number of bright data accessed
            cU = atomicAdd(&dataCounters[4],1);        // increase number of u data accessed
            
            LBb[cBright] = exp(sLBblock[0]) - exp(sLDblock[0]);

            ResampReg = 1 - exp(sLDblock[0])/exp(sLBblock[0]);
            if( ResampReg <= u[cU])
            {  
                z[bidx] = 0;
                atomicAdd(&dataCounters[2], 1);    // increase number of dark data
            }else{
                z[bidx] = 1;
                atomicAdd(&dataCounters[0], 1);             // increase number of bright data
            }
        }
    }
}

__global__ void lowerbound_kernel_lvl1(double *dotD, float *dotS, int N,
                                  double *LB, double *epsilon)
{
  // extern __shared__ double sarrayD1[];
  int tidx = threadIdx.x;
  int didx = blockIdx.x * blockDim.x + threadIdx.x;
  // double* sLBn = (double*) sarrayD1;
  // double* sEpsilon = (blockDim.x<=32) ? (double*) &sarrayD1[2*blockDim.x] : (double*) &sarrayD1[blockDim.x];
  __shared__ double sLBn[3072];
  __shared__ double sEpsilon[3072];

  if(didx < N)
  {
    sLBn[tidx] = -log(1 + exp(dotD[didx]));
    sEpsilon[tidx] = abs(exp(sLBn[tidx]) - (1 / (1 + expf(dotS[didx]))));
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

          if(sEpsilon[tidx] < sEpsilon[tidx+s])
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
  // extern __shared__ double sarrayD[];
  
  int tidx = threadIdx.x;
  int didx = blockIdx.x * blockDim.x + threadIdx.x;

  // double* sLBn = (double*) sarrayD;
  // double* sEpsilon = (blockDim.x<=32) ? (double*) &sarrayD[2*blockDim.x] : (double*) &sarrayD[blockDim.x];
  __shared__ double sLBn[3072];
  __shared__ double sEpsilon[3072];

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

          if(sEpsilon[tidx] < sEpsilon[tidx+s])
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

#endif // __MP_KERNEL_CU__