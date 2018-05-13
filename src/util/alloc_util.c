#include "alloc_util.h"

void malloc_data_vectors_cpu(data_str *data, mcmc_str mcin)
{
  data->data = NULL;
  data->labels = NULL;

  data->data = (double *) malloc(mcin.ddata * mcin.Nd * sizeof(double));
  if(data->data == NULL)
    fprintf(stderr, "ERROR: Data Memory allocation did not complete successfully!\n");
  
  data->labels = (int8_t *) malloc(mcin.Nd * sizeof(int8_t));  
  if(data->labels == NULL)
    fprintf(stderr, "ERROR: Labels Data Memory allocation did not complete successfully!\n");
}

void malloc_data_vectors_gpu(data_str *data, mcmc_str mcin)
{
  data->data = NULL;
  data->gpudata = NULL;
  data->labels = NULL;

  data->data = (double *) malloc(mcin.ddata * mcin.Nd * sizeof(double));
  if(data->data == NULL)
    fprintf(stderr, "ERROR: Data Memory allocation did not complete successfully!\n");
  
  data->gpudata = (double *) malloc(mcin.dmap * mcin.Nd * sizeof(double));
  if(data->gpudata == NULL)
    fprintf(stderr, "ERROR: GPU Data Memory allocation did not complete successfully!\n");  
  
  data->labels = (int8_t *) malloc(mcin.Nd * sizeof(int8_t));  
  if(data->labels == NULL)
    fprintf(stderr, "ERROR: Labels Data Memory allocation did not complete successfully!\n");  
}

void malloc_sample_vectors(mcmc_v_str *mcdata, mcmc_str mcin)
{
  mcdata->samples = NULL;
  mcdata->burn = NULL;
  mcdata->sample_means = NULL;

  mcdata->samples = (double *) malloc(mcin.ddata * mcin.Ns * sizeof(double));
  if(mcdata->samples == NULL)
    fprintf(stderr, "ERROR: Parameters Data Memory allocation did not complete successfully!\n");

  mcdata->burn = (double *) malloc(mcin.ddata * mcin.burnin * sizeof(double));
  if(mcdata->burn == NULL)
    fprintf(stderr, "ERROR: Burn Parameters Data Memory allocation did not complete successfully!\n");
}

void malloc_normalised_sample_vectors(mcmc_v_str *mcdata, mcmc_str mcin)
{
  mcdata->nsamples = NULL;
  mcdata->nburn = NULL;

  mcdata->nsamples = (double *) malloc(mcin.ddata * mcin.Ns * sizeof(double));
  if(mcdata->nsamples == NULL)
    fprintf(stderr, "ERROR: Normalised Parameters Data Memory allocation did not complete successfully!\n");

  mcdata->nburn = (double *) malloc(mcin.ddata * mcin.burnin * sizeof(double));
  if(mcdata->nburn == NULL)
    fprintf(stderr, "ERROR: Normalised Burned Parameters Data Memory allocation did not complete successfully!\n");

  mcdata->sample_means = (double *) malloc(mcin.ddata * sizeof(double));
  if(mcdata->sample_means == NULL)
    fprintf(stderr, "ERROR: Sample Means Memory allocation did not complete successfully!\n");
}

void malloc_autocorrelation_vectors(sec_v_str *secv, sec_str sec)
{
  secv->shift = NULL;
  secv->circ = NULL;

  if((sec.fauto == 1) || (sec.fauto == 3)){
    secv->shift = (double *) malloc(sec.lagidx * sizeof(double));
    if(secv->shift == NULL)
      fprintf(stderr, "ERROR: Shift Autocorrelation Data Memory allocation did not complete successfully!\n");    
  }

  if((sec.fauto == 2) || (sec.fauto == 3)){  
    secv->circ = (double *) malloc(sec.lagidx * sizeof(double));
    if(secv->circ == NULL)
      fprintf(stderr, "ERROR: Circular Autocorrelation Data Memory allocation did not complete successfully!\n");  
  }    
}

void malloc_mcmc_vectors_cpu(mcmc_int_v *mclocv, mcmc_str mcin)
{
  mclocv->proposed = NULL;
  mclocv->current = NULL;

  mclocv->proposed = (double *) malloc(mcin.ddata * sizeof(double));
  if(mclocv->proposed == NULL)
    fprintf(stderr, "ERROR: Proposed Samples Memory allocation did not complete successfully!\n");
  
  mclocv->current = (double *) malloc(mcin.ddata * sizeof(double));  
  if(mclocv->current == NULL)
    fprintf(stderr, "ERROR: Current Samples Memory allocation did not complete successfully!\n");
}

void malloc_mcmc_vectors_gpu(mcmc_int_v *mclocv, mcmc_str mcin)
{
  mclocv->proposed = NULL;
  mclocv->current = NULL;

  mclocv->proposed = (double *) malloc(mcin.dmap * sizeof(double));
  if(mclocv->proposed == NULL)
    fprintf(stderr, "ERROR: Proposed Samples Memory allocation did not complete successfully!\n");

  mclocv->current = (double *) malloc(mcin.dmap * sizeof(double));  
  if(mclocv->current == NULL)
    fprintf(stderr, "ERROR: Current Samples Memory allocation did not complete successfully!\n");

  // zero pad the vectors
  int idx;
  for(idx=0; idx<mcin.dmap; idx++){
    mclocv->current[idx] = 0;
    mclocv->proposed[idx] = 0;
  }
}

void free_data_vectors_cpu(data_str data)
{
  free(data.data);
  free(data.labels);
}

void free_data_vectors_gpu(data_str data)
{
  free(data.data);
  free(data.gpudata);
  free(data.labels);
}

void free_sample_vectors(mcmc_v_str mcdata)
{
  free(mcdata.samples);
  free(mcdata.burn); 
}

void free_norm_sample_vectors(mcmc_v_str mcdata)
{
  free(mcdata.nsamples);
  free(mcdata.nburn);  
  free(mcdata.sample_means); 
}

void free_autocorrelation_vectors(sec_v_str secv, sec_str sec)
{
  if((sec.fauto == 1) || (sec.fauto == 3)){
    free(secv.shift);
  }
  if((sec.fauto == 2) || (sec.fauto == 3)){  
    free(secv.circ);
  }  
}

void free_mcmc_vectors_cpu(mcmc_int_v mclocv)
{
  free(mclocv.proposed);
  free(mclocv.current);
}

void free_mcmc_vectors_gpu(mcmc_int_v mclocv)
{
  free(mclocv.proposed);
  free(mclocv.current); 
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