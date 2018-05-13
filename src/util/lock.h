#ifndef __LOCK_H__
#define __LOCK_H__

struct Lock{
  int *mutex;
  
  Lock(void){
      int state = 0;
      cudaMalloc((void **)& mutex, sizeof(int));
      cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
  }

  ~Lock(void){
      cudaFree(mutex);
  }

  __device__ void lock(void){
      while(atomicCAS(mutex, 0, 1) != 0);
  }

  __device__ void unlock(void){
      atomicExch(mutex, 1);
  }
};

#endif //__LOCK_H__