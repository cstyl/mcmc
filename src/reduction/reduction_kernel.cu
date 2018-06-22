/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <stdio.h>
#include <cooperative_groups.h>

#define FULL_MASK 0xffffffff

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

template <class T>
__global__ void
reduceRegr0(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? -log(1+exp(g_idata[i])) : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/* This version uses contiguous threads, but its interleaved
   addressing results in many shared memory bank conflicts.
*/
template <class T>
__global__ void
reduceRegr1(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? -log(1+exp(g_idata[i])) : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses sequential addressing -- no divergence or bank conflicts.
*/
template <class T>
__global__ void
reduceRegr2(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? -log(1+exp(g_idata[i])) : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

/* This reduction interleaves which threads are active by using the modulo
   operator.  This operator is very expensive on GPUs, and the interleaved
   inactivity means that no whole warps are active, which is also very
   inefficient */
template <class T>
__global__ void
reduce0(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/* This version uses contiguous threads, but its interleaved
   addressing results in many shared memory bank conflicts.
*/
template <class T>
__global__ void
reduce1(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses sequential addressing -- no divergence or bank conflicts.
*/
template <class T>
__global__ void
reduce2(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T>
__global__ void
reduceLog0(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? log(g_idata[i]) : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        // modulo arithmetic is slow!
        if ((tid % (2*s)) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/* This version uses contiguous threads, but its interleaved
   addressing results in many shared memory bank conflicts.
*/
template <class T>
__global__ void
reduceLog1(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? log(g_idata[i]) : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version uses sequential addressing -- no divergence or bank conflicts.
*/
template <class T>
__global__ void
reduceLog2(T *g_idata, T *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T *sdata = SharedMemory<T>();

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? log(g_idata[i]) : 0;

    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
void reduceSum_d(int size, int threads, int blocks,
       int whichKernel, double *d_idata, double *d_odata,
       int type)  // 0=transform & addition, 1=addition, 2=log_addition
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(double) : threads * sizeof(double);

    switch(type)
    {   
        case 0:
        // choose which of the optimized versions of reduction to launch
            switch (whichKernel)
            {
                case 0:
                    reduceRegr0<double><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 1:
                    reduceRegr1<double><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 2:
                    reduceRegr2<double><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

            }
            break;
        case 1:
            // choose which of the optimized versions of reduction to launch
            switch (whichKernel)
            {
                case 0:
                    reduce0<double><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 1:
                    reduce1<double><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 2:
                    reduce2<double><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;


            }
            break; 
        case 2:
            // choose which of the optimized versions of reduction to launch
            switch (whichKernel)
            {
                case 0:
                    reduceLog0<double><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 1:
                    reduceLog1<double><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 2:
                    reduceLog2<double><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

            }
            break;    
    }
    
}


void reduceSum_f(int size, int threads, int blocks,
       int whichKernel, float *d_idata, float *d_odata,
       int type)  // 0=transform & addition, 1=addition, 2=mult
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    switch(type)
    {   
        case 0:
        // choose which of the optimized versions of reduction to launch
            switch (whichKernel)
            {
                case 0:
                    reduceRegr0<float><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 1:
                    reduceRegr1<float><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 2:
                    reduceRegr2<float><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

            }
            break;
        case 1:
            // choose which of the optimized versions of reduction to launch
            switch (whichKernel)
            {
                case 0:
                    reduce0<float><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 1:
                    reduce1<float><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 2:
                    reduce2<float><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

            }
            break; 
        case 2:
            // choose which of the optimized versions of reduction to launch
            switch (whichKernel)
            {
                case 0:
                    reduceLog0<float><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 1:
                    reduceLog1<float><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

                case 2:
                    reduceLog2<float><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                    break;

            }
            break;    
    }
    
}
#endif // _REDUCE_KERNEL_H_
