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

#ifndef __REDUCTION_H__
#define __REDUCTION_H__

#if defined (__cplusplus)
extern "C" {
#endif

bool isPow2(unsigned int x);

#if defined (__cplusplus)
}
#endif

void reduceSum_d(int size, int threads, int blocks,
       int whichKernel, double *d_idata, double *d_odata,
       int type);

void reduceSum_f(int size, int threads, int blocks,
       int whichKernel, float *d_idata, float *d_odata,
       int type);
#endif
