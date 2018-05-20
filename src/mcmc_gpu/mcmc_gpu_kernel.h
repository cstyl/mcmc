#ifndef __GPU_MCMC_KERNEL_H__
#define __GPU_MCMC_KERNEL_H__

#if defined (__cplusplus)
extern "C" {
#endif

bool isPow2(unsigned int x);

void reductiond(int size, int threads, int blocks, int kernel, 
                bool firstIt, double *in_d, double *out_d);

#if defined (__cplusplus)
}
#endif

#endif //__GPU_MCMC_KERNEL_H__