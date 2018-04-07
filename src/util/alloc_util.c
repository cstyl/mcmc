#include "alloc_util.h"

void malloc_data_vectors_cpu(double **data, double **labels, 
                        int data_dim, int datapoints)
{
  *data = NULL;
  *labels = NULL;

  *data = (double *) malloc(data_dim * datapoints * sizeof(double));
  if(*data == NULL)
    fprintf(stderr, "ERROR: Data Memory allocation did not complete successfully!\n");
  
  *labels = (double *) malloc(datapoints * sizeof(double));  
  if(*labels == NULL)
    fprintf(stderr, "ERROR: Labels Data Memory allocation did not complete successfully!\n");
}

// void data_vectors_gpu(int data_dim, int gpu_data_dim, int datapoints)
// {
//   x = (double *) malloc(data_dim * datapoints * sizeof(double));
//   gpu_x = (double *) malloc(gpu_data_dim * datapoints * sizeof(double));
//   y = (double *) malloc(datapoints * sizeof(double));  
// }

void malloc_sample_vectors(double **parameters, double **burn_parameters, 
                    int data_dim, int samples, int burn_samples)
{
  *parameters = NULL;
  *burn_parameters = NULL;

  *parameters = (double *) malloc(data_dim * samples * sizeof(double));
  if(*parameters == NULL)
    fprintf(stderr, "ERROR: Parameters Data Memory allocation did not complete successfully!\n");

  *burn_parameters = (double *) malloc(data_dim * burn_samples * sizeof(double));
  if(*burn_parameters == NULL)
    fprintf(stderr, "ERROR: Burn Parameters Data Memory allocation did not complete successfully!\n");
}

void malloc_normalised_sample_vectors(double **norm_par, double **norm_burn_par, 
                              int data_dim, int samples, int burn_samples)
{
  *norm_par = NULL;
  *norm_burn_par = NULL;

  *norm_par = (double *) malloc(data_dim * samples * sizeof(double));
  if(*norm_par == NULL)
    fprintf(stderr, "ERROR: Normalised Parameters Data Memory allocation did not complete successfully!\n");

  *norm_burn_par =  (double *) malloc(data_dim * burn_samples * sizeof(double));
  if(*norm_burn_par == NULL)
    fprintf(stderr, "ERROR: Normalised Burned Parameters Data Memory allocation did not complete successfully!\n");
}

void malloc_autocorrelation_vectors(double **auto_shift, double **auto_circ,
                              int auto_case, int lag)
{
  *auto_shift = NULL;
  *auto_circ = NULL;

  if((auto_case == 1) || (auto_case == 3)){
    *auto_shift = (double *) malloc(lag * sizeof(double));
    if(*auto_shift == NULL)
      fprintf(stderr, "ERROR: Shift Autocorrelation Data Memory allocation did not complete successfully!\n");    
  }

  if((auto_case == 2) || (auto_case == 3)){  
    *auto_circ = (double *) malloc(lag * sizeof(double));
    if(*auto_circ == NULL)
      fprintf(stderr, "ERROR: Circular Autocorrelation Data Memory allocation did not complete successfully!\n");  
  }  
}

void free_data_vectors_cpu(double *data, double *labels)
{
  free(data);
  free(labels);
}

// void free_data_vectors_gpu()
// {
//   free(x);
//   free(gpu_x);
//   free(y);
// }

void free_sample_vectors(double *parameters, double *burn_parameters)
{
  free(parameters);
  free(burn_parameters);  
}

void free_norm_sample_vectors(double *norm_par, double *norm_burn_par)
{
  free(norm_par);
  free(norm_burn_par);
}

void free_autocorrelation_vectors(double *auto_shift, double *auto_circ,
                                  int auto_case)
{
  if((auto_case == 1) || (auto_case == 3)){
    free(auto_shift);
  }
  if((auto_case == 2) || (auto_case == 3)){  
    free(auto_circ);
  }  
}

void init_rng(gsl_rng **r)
{
  const gsl_rng_type * T;

  gsl_rng_env_setup();

  T = gsl_rng_default;
  *r = gsl_rng_alloc(T);
  if(*r == NULL)
    fprintf(stderr, "ERROR: RNG Allocation did not complete successfully!\n");
}

void free_rng(gsl_rng *r)
{
  gsl_rng_free(r); 
}