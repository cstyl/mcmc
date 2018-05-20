#ifndef __UTIL_GPU_H__
#define __UTIL_GPU_H__

#include "resources.h"

#if defined (__cplusplus)
extern "C" {
#endif

int nextPow2(int d_data);
void getBlocksAndThreads(int kernel, int n, int maxBlocks, int maxThreads, int *blocks, int *threads);

#if defined (__cplusplus)
}
#endif

#endif //   __UTIL_GPU_H__