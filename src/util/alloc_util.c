#include "cpu_host.h"
#include "alloc_util.h"

void data_vectors_cpu(int data_dim, int datapoints)
{
  x = (double *) malloc(data_dim * datapoints * sizeof(double));
  y = (double *) malloc(datapoints * sizeof(double));  
}

// void data_vectors_gpu(int data_dim, int gpu_data_dim, int datapoints)
// {
//   x = (double *) malloc(data_dim * datapoints * sizeof(double));
//   gpu_x = (double *) malloc(gpu_data_dim * datapoints * sizeof(double));
//   y = (double *) malloc(datapoints * sizeof(double));  
// }

void sample_vectors(int data_dim, int samples, int burn_samples)
{
  sample_matrix = (double *) malloc(data_dim * samples * sizeof(double));
  burn_matrix =  (double *) malloc(data_dim * burn_samples * sizeof(double));
}

void normalised_sample_vectors(int data_dim, int samples, int burn_samples)
{
  norm_sample_matrix = (double *) malloc(data_dim * samples * sizeof(double));
  norm_burn_matrix =  (double *) malloc(data_dim * burn_samples * sizeof(double));
}

void autocorrelation_vectors(int auto_case, int lag)
{
  if((auto_case == 1) || (auto_case == 3)){
    autocorrelation_shift = (double *) malloc(lag * sizeof(double));
  }
  if((auto_case == 2) || (auto_case == 3)){  
    autocorrelation_circ = (double *) malloc(lag * sizeof(double));
  }  
}

void free_data_vectors_cpu()
{
  free(x);
  free(y);
}

// void free_data_vectors_gpu()
// {
//   free(x);
//   free(gpu_x);
//   free(y);
// }

void free_sample_vectors()
{
  free(sample_matrix);
  free(burn_matrix);  
}

void free_norm_sample_vectors()
{
  free(norm_sample_matrix);
  free(norm_burn_matrix);
}

void free_autocorrelation_vectors(int auto_case)
{
  if((auto_case == 1) || (auto_case == 3)){
    free(autocorrelation_shift);
  }
  if((auto_case == 2) || (auto_case == 3)){  
    free(autocorrelation_circ);
  }  
}

void init_rng()
{
  const gsl_rng_type * T;

  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc(T);
}

void free_rng()
{
  gsl_rng_free(r); 
}