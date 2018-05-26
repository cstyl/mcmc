#ifndef __LHOOD_MP_KERNEL_CU__
#define __LHOOD_MP_KERNEL_CU__

#include "lhood_mp_kernel.h"

// sequential addressing double precision
__global__ void brightLhood(double *din_d, float *din_f,
                            double *samples_d, float *samples_f,
                            int *z_index, 
                            unsigned int dim,
                            unsigned int Nb, unsigned int N, // N is total threads to be executed
                            double *cuLhood, float *cuLhoodf,
                            double *resample
                            )
{
  int zidx = blockIdx.x * blockDim.x + threadIdx.x;
  int data_idx = z_index[zidx];

  if(zidx < N)
  {  
    double LDn = 1;
    float  LCn = 1;

    LDn /= 1 + exp(-cublasDdot(handle, dim, &din_d[data_idx], 1, samples_d, 1)); 
    LCn /= 1 + exp(-cublasSdot(handle, dim, &din_f[data_idx], 1, samples_f, 1)); 
    cudaDeviceSynchronize();

    if zidx < Nb
    {
      cuLhood[zidx] = LDn - LCn;
      resample[zidx] = 1 - LCn/LDn;
    }else{
      cuLhoodf[zidx] = LCn;
      resample[zidx+N] = 1 - LCn/LDn;
    }
  }
}

__global__ void darkLhood(float *din_f, float *samples_f,
                          int *z_index,
                          unsigned int dim, 
                          unsigned int Nc, unsigned int Nb,
                          unsigned int N,  // total threads
                          float *cuLhoodf
                          )
{
  int zidx = blockIdx.x * blockDim.x + threadIdx.x; 
  int offset = Nb + Nc;                                                   
  int data_idx = z_index[zidx+offset];

  if(zidx < N)
  {  
    float  LCn = 1;
   
    LCn /= 1 + exp(-cublasSdot(handle, dim, &din_f[data_idx], 1, samples_f, 1)); 
    cudaDeviceSynchronize();

    cuLhoodf[zidx+offset] = LCn;
  }
}


void brightL(int threads, int blocks, dev_v_str d, mcmc_str mcin)
{
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  brightLhood<<< dimGridm, dimBlock >>>(d.data, d.dataf, d.samples, d.samplesf, d.zidx,
                                        mcin.ddata, mcin.bright, mcin.bright+mcin.cand,
                                        d.cuLhood, d.cuLhoodf, d.resample);

}

void darkL(int threads, int blocks, dev_v_str d, mcmc_str mcin)
{
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  darkLhood<<< dimGridm, dimBlock >>>(d.dataf, d.samplesf, d.zidx, mcin.ddata, 
                                      mcin.cand, mcin.bright, mcin.dark,
                                      d.cuLhoodf);
}



#endif  //__LHOOD_MP_KERNEL_CU__