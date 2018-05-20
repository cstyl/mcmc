#include "gpu_util.h"

#define MIN(x,y) ((x < y) ? x : y)

int nextPow2(int d_data){
  int ctr = 0;

  while(d_data!=0){
    d_data = d_data/2;
    ctr++;
  }

  return pow(2,ctr);
}

void getBlocksAndThreads(int kernel, int n, int maxBlocks, int maxThreads, int *blocks, int *threads)
{
    int lthreads, lblocks;

    if(kernel<1)
    {
        // if data less than the # of threads per block make threads per block the next power of 2
        // otherwise use the preset maximum threads
        lthreads = (n < maxThreads) ? nextPow2(n) : maxThreads;  
        lblocks = (n + lthreads - 1) / lthreads;
    }else{
        lthreads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        lblocks = (n + (lthreads * 2 - 1)) / (lthreads * 2);
    }

    if(kernel == 4) lblocks = MIN(maxBlocks, lblocks);

    *threads = lthreads;
    *blocks = lblocks;
}