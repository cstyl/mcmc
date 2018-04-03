#ifndef __ALLOC_UTIL_H__
#define __ALLOC_UTIL_H__

void data_vectors_cpu(int data_dim, int datapoints);
// void data_vectors_gpu(int data_dim, int gpu_data_dim, int datapoints);
void sample_vectors(int data_dim, int samples, int burn_samples);
void normalised_sample_vectors(int data_dim, int samples, int burn_samples);

void free_data_vectors_cpu();
// void free_data_vectors_gpu();
void free_sample_vectors();
void free_norm_sample_vectors();

void autocorrelation_vectors(int auto_case, int lag);
void free_autocorrelation_vectors(int auto_case);

void init_rng();
void free_rng();


#endif  //__ALLOC_UTIL_H__