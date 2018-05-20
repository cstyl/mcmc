#include "mcmc_gpu_kernel.h"
#include <cooperative_groups.h>
#define FULL_MASK 0xffffffff

namespace cg = cooperative_groups;

bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

// sequential addressing double precision
__global__ void reductiond0( double *din, // input data: matrix to vector mul
                             double *dout,  // output data: vector of likelihoods sum
                             unsigned int n, // total threads
                             bool firstIt  // first iteration of the reduction flag
                            )
{
  extern __shared__ double sdata[];

  int tidx = threadIdx.x;
  int didx = blockIdx.x * blockDim.x + threadIdx.x;;

  // loads a single dot product to shared memory if this is the first time the kernel runs
  // otherwise loads tha accumulated data for previous run of reduction
  if(firstIt)
    sdata[tidx] = (didx < n) ? -log(1+exp(din[didx])) : 0;
  else
    sdata[tidx] = (didx < n) ? din[didx] : 0;

  __syncthreads();

  // perform reduction using shared memory
  unsigned int i = blockDim.x/2;
  for (i=blockDim.x/2; i>0; i>>=1){
    if (tidx < i)
      sdata[tidx] += sdata[tidx + i];
    __syncthreads();
  }

  // write result for this block back to global mem
  if (tidx == 0) dout[blockIdx.x] = sdata[0]; 
}

// perform first level of reduction while reading from global memory
// store the result to shared memory
__global__ void reductiond1( double *din, // input data: matrix to vector mul
                             double *dout,  // output data: vector of likelihoods sum
                             unsigned int n, // total threads
                             bool firstIt  // iteration of the reduction
                            )
{
  extern __shared__ double sdata[];
  double redsum;   

  int tidx = threadIdx.x;
  int didx = blockIdx.x * (blockDim.x*2) + threadIdx.x; // doubling the block size

  // loads a single dot product to shared memory if this is the first time the kernel runs
  // otherwise loads tha accumulated data for previous run of reduction
  if(firstIt){  
    redsum = (didx < n) ? -log(1+exp(din[didx])) : 0;
    
    if(didx + blockDim.x < n)
      redsum += -log(1+exp(din[didx+blockDim.x]));
  
  }else{
    redsum = (didx < n) ? din[didx] : 0;
    
    if(didx + blockDim.x < n)
      redsum += din[didx+blockDim.x];
  }
  
  sdata[tidx] = redsum;   // store the result in share memory
  __syncthreads();

  // perform reduction using shared memory
  unsigned int i;
  for(i=blockDim.x/2; i>0; i>>=1) 
  {
    if (tidx < i){
      sdata[tidx] = redsum = redsum + sdata[tidx + i];
    }
    __syncthreads();
  }
  // write result for this block back to global mem
  if (tidx == 0) dout[blockIdx.x] = redsum; 
}

__global__ void reductiond2( double *din, // input data: matrix to vector mul
                             double *dout,  // output data: vector of likelihoods sum
                             unsigned int n, // total threads
                             bool firstIt,  // iteration of the reduction
                             unsigned int blocksz
                            )
{
  extern __shared__ double sdata[];
  
  unsigned int blockSize = blocksz;
  double redsum;

  int tidx = threadIdx.x;
  int didx = blockIdx.x * (blockDim.x*2) + threadIdx.x; // doubling the block size

  // loads a single dot product to shared memory if this is the first time the kernel runs
  // otherwise loads tha accumulated data for previous run of reduction
  if(firstIt){  
    redsum = (didx < n) ? -log(1+exp(din[didx])) : 0;
    
    if(didx + blockSize < n)
      redsum += -log(1+exp(din[didx+blockSize]));
  
  }else{
    redsum = (didx < n) ? din[didx] : 0;
    
    if(didx + blockSize < n)
      redsum += din[didx+blockSize];
  }
  
  sdata[tidx] = redsum;
  __syncthreads();

  // perform reduction using shared memory
  unsigned int i;
  for(i=blockDim.x/2; i>32; i>>=1) {
    if (tidx < i){
      sdata[tidx] = redsum = redsum + sdata[tidx + i];
    }
    __syncthreads(); 
  }

#if (__CUDA_ARCH__ >= 300)
  if(tidx < 32)
  {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >=  64) redsum += sdata[tidx + 32];
    // Reduce final warp using shuffle
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
    {
      redsum += __shfl_down_sync(FULL_MASK, redsum, offset);
    }
  }
#else 
  if((blockSize >= 64) && (tidx < 32)){ sdata[tidx] = redsum = redsum + sdata[tidx + 32]; }
  __syncthreads();

  if((blockSize >= 32) && (tidx < 16)){ sdata[tidx] = redsum = redsum + sdata[tidx + 16]; }
  __syncthreads();

  if((blockSize >= 16) && (tidx <  8)){ sdata[tidx] = redsum = redsum + sdata[tidx +  8]; }
  __syncthreads();

  if((blockSize >=  8) && (tidx <  4)){ sdata[tidx] = redsum = redsum + sdata[tidx +  4]; }
  __syncthreads();

  if((blockSize >=  4) && (tidx <  2)){ sdata[tidx] = redsum = redsum + sdata[tidx +  2]; }
  __syncthreads();

  if((blockSize >=  2) && (tidx <  1)){ sdata[tidx] = redsum = redsum + sdata[tidx +  1]; }
  __syncthreads();
#endif
  // write result for this block back to global mem
  if (tidx == 0) dout[blockIdx.x] = redsum; 
}

__global__ void reductiond3( double *din, // input data: matrix to vector mul
                             double *dout,  // output data: vector of likelihoods sum
                             unsigned int n, // total threads
                             bool firstIt,  // iteration of the reduction
                             unsigned int blocksz
                            )
{
  extern __shared__ double sdata[];

  double redsum;   
  unsigned int blockSize = blocksz;

  int tidx = threadIdx.x;
  int didx = blockIdx.x * (blockDim.x*2) + threadIdx.x; // doubling the block size

  // loads a single dot product to shared memory if this is the first time the kernel runs
  // otherwise loads tha accumulated data for previous run of reduction
  if(firstIt){  
    redsum = (didx < n) ? -log(1+exp(din[didx])) : 0;
    
    if(didx + blockSize < n)
      redsum += -log(1+exp(din[didx+blockSize]));
  
  }else{
    redsum = (didx < n) ? din[didx] : 0;
    
    if(didx + blockSize < n)
      redsum += din[didx+blockSize];
  }
  
  sdata[tidx] = redsum;   // store the result in share memory
  __syncthreads();

  // perform reduction using shared memory
  if((blockSize >= 1024) && (tidx < 512)){ sdata[tidx] = redsum = redsum + sdata[tidx + 512]; }
  __syncthreads();
  
  if((blockSize >=  512) && (tidx < 256)){ sdata[tidx] = redsum = redsum + sdata[tidx + 256]; }
  __syncthreads();
  
  if((blockSize >=  256) && (tidx < 128)){ sdata[tidx] = redsum = redsum + sdata[tidx + 128]; }
  __syncthreads();
  
  if((blockSize >=  128) && (tidx <  64)){ sdata[tidx] = redsum = redsum + sdata[tidx +  64]; }
  __syncthreads();

#if (__CUDA_ARCH__ >= 300)
  if(tidx < 32)
  {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >=  64) redsum += sdata[tidx + 32];
    // Reduce final warp using shuffle
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
    {
      redsum += __shfl_down_sync(FULL_MASK, redsum, offset);
    }
  }
#else  
  if((blockSize >= 64) && (tidx < 32)){ sdata[tidx] = redsum = redsum + sdata[tidx + 32]; }
  __syncthreads();

  if((blockSize >= 32) && (tidx < 16)){ sdata[tidx] = redsum = redsum + sdata[tidx + 16]; }
  __syncthreads();

  if((blockSize >= 16) && (tidx <  8)){ sdata[tidx] = redsum = redsum + sdata[tidx +  8]; }
  __syncthreads();

  if((blockSize >=  8) && (tidx <  4)){ sdata[tidx] = redsum = redsum + sdata[tidx +  4]; }
  __syncthreads();

  if((blockSize >=  4) && (tidx <  2)){ sdata[tidx] = redsum = redsum + sdata[tidx +  2]; }
  __syncthreads();

  if((blockSize >=  2) && (tidx <  1)){ sdata[tidx] = redsum = redsum + sdata[tidx +  1]; }
  __syncthreads();
#endif
  // write result for this block back to global mem
  if (tidx == 0) dout[blockIdx.x] = redsum; 
}

__global__ void reductiond4( double *din, // input data: matrix to vector mul
                             double *dout,  // output data: vector of likelihoods sum
                             unsigned int n, // total threads
                             bool firstIt,  // iteration of the reduction
                             unsigned int blocksz,
                             bool Pow2
                            )
{
  extern __shared__ double sdata[];

  double redsum = 0;   
  unsigned int blockSize = blocksz;

  unsigned int tidx = threadIdx.x;
  unsigned int didx = blockIdx.x * (blockSize*2) + threadIdx.x; // doubling the block size
  unsigned int gridSize = blockSize*2*gridDim.x;

  // loads a single dot product to shared memory if this is the first time the kernel runs
  // otherwise loads tha accumulated data for previous run of reduction
  if(firstIt){  
    while(didx < n)
    {
      redsum += -log(1+exp(din[didx]));

      if(Pow2 || (didx + blockSize < n))
        redsum += -log(1+exp(din[didx+blockSize]));

      didx += gridSize;
    }
  }else{
    while(didx < n)
    {
      redsum += din[didx];

      if(Pow2 || (didx + blockSize < n))
        redsum += din[didx+blockSize];

      didx += gridSize;
    }
  }
  
  sdata[tidx] = redsum;   // store the result in share memory
  __syncthreads();

  // perform reduction using shared memory
  if((blockSize >= 1024) && (tidx < 512)){ sdata[tidx] = redsum = redsum + sdata[tidx + 512]; }
  __syncthreads();
  
  if((blockSize >=  512) && (tidx < 256)){ sdata[tidx] = redsum = redsum + sdata[tidx + 256]; }
  __syncthreads();
  
  if((blockSize >=  256) && (tidx < 128)){ sdata[tidx] = redsum = redsum + sdata[tidx + 128]; }
  __syncthreads();
  
  if((blockSize >=  128) && (tidx <  64)){ sdata[tidx] = redsum = redsum + sdata[tidx +  64]; }
  __syncthreads();

#if (__CUDA_ARCH__ >= 300)
  if(tidx < 32)
  {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >=  64) redsum += sdata[tidx + 32];
    // Reduce final warp using shuffle
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
    {
      redsum += __shfl_down_sync(FULL_MASK, redsum, offset);
    }
  }
#else  
  if((blockSize >= 64) && (tidx < 32)){ sdata[tidx] = redsum = redsum + sdata[tidx + 32]; }
  __syncthreads();

  if((blockSize >= 32) && (tidx < 16)){ sdata[tidx] = redsum = redsum + sdata[tidx + 16]; }
  __syncthreads();

  if((blockSize >= 16) && (tidx <  8)){ sdata[tidx] = redsum = redsum + sdata[tidx +  8]; }
  __syncthreads();

  if((blockSize >=  8) && (tidx <  4)){ sdata[tidx] = redsum = redsum + sdata[tidx +  4]; }
  __syncthreads();

  if((blockSize >=  4) && (tidx <  2)){ sdata[tidx] = redsum = redsum + sdata[tidx +  2]; }
  __syncthreads();

  if((blockSize >=  2) && (tidx <  1)){ sdata[tidx] = redsum = redsum + sdata[tidx +  1]; }
  __syncthreads();
#endif
  // write result for this block back to global mem
  if (tidx == 0) dout[blockIdx.x] = redsum; 
}


void reductiond(int size, int threads, int blocks, int kernel, 
                bool firstIt, double *in_d, double *out_d)
{
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);
  int smemSize;
  
  if(threads <= 32)
    smemSize = 2 * threads * sizeof(double);
  else
    smemSize = threads * sizeof(double);

  switch (kernel)
  {
    case 0: // sequential addressing
      reductiond0<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt);
      break;
    case 1: // first reduction while loading from memory
      reductiond1<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt);
      break;
    case 2: // unroll final warp
      switch(threads)
      {
        case 1024:
          reductiond2<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt, 1024);
          break;
        case  512:
          reductiond2<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,  512);
          break;
        case  256:
          reductiond2<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,  256);
          break;
        case  128:
          reductiond2<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,  128);
          break;
        case   64:
          reductiond2<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,   64);
          break;
        case   32:
          reductiond2<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,   32);
          break;
        case   16:
          reductiond2<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,   16);
          break;
        case    8:
          reductiond2<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,    8);
          break;
        case    4:
          reductiond2<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,    4);
          break;
        case    2:
          reductiond2<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,    2);
          break;
        case    1:
          reductiond2<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,    1);
          break;
      }
      break;
    case 3: // unroll final warp
      switch(threads)
      {
        case 1024:
          reductiond3<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt, 1024);
          break;
        case  512:
          reductiond3<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,  512);
          break;
        case  256:
          reductiond3<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,  256);
          break;
        case  128:
          reductiond3<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,  128);
          break;
        case   64:
          reductiond3<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,   64);
          break;
        case   32:
          reductiond3<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,   32);
          break;
        case   16:
          reductiond3<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,   16);
          break;
        case    8:
          reductiond3<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,    8);
          break;
        case    4:
          reductiond3<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,    4);
          break;
        case    2:
          reductiond3<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,    2);
          break;
        case    1:
          reductiond3<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,    1);
          break;
      }
      break;
    case 4:
    default:
      if(isPow2(size))
      {
        switch(threads)
        {
          case 1024:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt, 1024, true);
            break;
          case  512:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,  512, true);
            break;
          case  256:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,  256, true);
            break;
          case  128:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,  128, true);
            break;
          case   64:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,   64, true);
            break;
          case   32:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,   32, true);
            break;
          case   16:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,   16, true);
            break;
          case    8:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,    8, true);
            break;
          case    4:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,    4, true);
            break;
          case    2:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,    2, true);
            break;
          case    1:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,    1, true);
            break;
        }
      }else{
        switch(threads)
        {
          case 1024:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt, 1024, false);
            break;
          case  512:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,  512, false);
            break;
          case  256:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,  256, false);
            break;
          case  128:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,  128, false);
            break;
          case   64:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,   64, false);
            break;
          case   32:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,   32, false);
            break;
          case   16:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,   16, false);
            break;
          case    8:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,    8, false);
            break;
          case    4:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,    4, false);
            break;
          case    2:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,    2, false);
            break;
          case    1:
            reductiond4<<< dimGrid, dimBlock, smemSize >>>(in_d, out_d, size, firstIt,    1, false);
            break;
        }
      }
      break;
  }
}
