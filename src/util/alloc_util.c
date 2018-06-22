#ifndef __ALLOC_UTIL_C__
#define __ALLOC_UTIL_C__

#include "alloc_util.h"

void malloc_data_vectors(data_str *data, mcmc_str mcin)
{
  data->data = NULL;
  data->mvout = NULL;  


  data->data = (double *) malloc(mcin.ddata * mcin.Nd * sizeof(double));
  if(data->data == NULL)
    fprintf(stderr, "ERROR: Data Memory allocation did not complete successfully!\n");

  data->mvout = (double *) malloc(mcin.Nd * sizeof(double));  
  if(data->mvout == NULL)
    fprintf(stderr, "ERROR: MVout Data Memory allocation did not complete successfully!\n"); 
}

void malloc_data_vectors_sp(data_str *data, mcmc_str mcin)
{
  data->mvoutf = NULL;
  data->dataf = NULL;

  data->dataf = (float *) malloc(mcin.ddata * mcin.Nd * sizeof(float));
  if(data->dataf == NULL)
    fprintf(stderr, "ERROR: Dataf Memory allocation did not complete successfully!\n");

  data->mvoutf = (float *) malloc(mcin.Nd * sizeof(float));  
  if(data->mvoutf == NULL)
    fprintf(stderr, "ERROR: MVoutf Data Memory allocation did not complete successfully!\n"); 
}

void malloc_data_vectors_mp(data_str *data, mcmc_str mcin)
{
  data->data = NULL;
  data->dataf = NULL;

  data->data = (double *) malloc(mcin.ddata * mcin.Nd * sizeof(double));
  if(data->data == NULL)
    fprintf(stderr, "ERROR: Data Memory allocation did not complete successfully!\n");    

  data->dataf = (float *) malloc(mcin.ddata * mcin.Nd * sizeof(float));
  if(data->dataf == NULL)
    fprintf(stderr, "ERROR: Single Precision Data Memory allocation did not complete successfully!\n");
}

void malloc_sample_vectors(mcmc_v_str *mcdata, mcmc_str mcin)
{
  mcdata->samples = NULL;
  mcdata->burn = NULL;

  mcdata->nsamples = NULL;
  mcdata->nburn = NULL;

  mcdata->sample_means = NULL;
  
  mcdata->samples = (double *) malloc(mcin.ddata * mcin.Ns * sizeof(double));
  if(mcdata->samples == NULL)
    fprintf(stderr, "ERROR: Parameters Data Memory allocation did not complete successfully!\n");

  mcdata->burn = (double *) malloc(mcin.ddata * mcin.burnin * sizeof(double));
  if(mcdata->burn == NULL)
    fprintf(stderr, "ERROR: Burn Parameters Data Memory allocation did not complete successfully!\n");
  
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

void malloc_sample_vectors_sp(mcmc_v_str *mcdata, mcmc_str mcin)
{
  mcdata->samplesf = NULL;
  mcdata->burnf = NULL;

  mcdata->nsamplesf = NULL;
  mcdata->nburnf = NULL;

  mcdata->sample_meansf = NULL;
  
  mcdata->samplesf = (float *) malloc(mcin.ddata * mcin.Ns * sizeof(float));
  if(mcdata->samplesf == NULL)
    fprintf(stderr, "ERROR: Parameters Data Memory allocation did not complete successfully!\n");

  mcdata->burnf = (float *) malloc(mcin.ddata * mcin.burnin * sizeof(float));
  if(mcdata->burnf == NULL)
    fprintf(stderr, "ERROR: Burn Parameters Data Memory allocation did not complete successfully!\n");
  
  mcdata->nsamplesf = (float *) malloc(mcin.ddata * mcin.Ns * sizeof(float));
  if(mcdata->nsamplesf == NULL)
    fprintf(stderr, "ERROR: Normalised Parameters Data Memory allocation did not complete successfully!\n");

  mcdata->nburnf = (float *) malloc(mcin.ddata * mcin.burnin * sizeof(float));
  if(mcdata->nburnf == NULL)
    fprintf(stderr, "ERROR: Normalised Burned Parameters Data Memory allocation did not complete successfully!\n");

  mcdata->sample_meansf = (float *) malloc(mcin.ddata * sizeof(float));
  if(mcdata->sample_meansf == NULL)
    fprintf(stderr, "ERROR: Sample Means Memory allocation did not complete successfully!\n");
}

void malloc_autocorrelation_vectors(sec_v_str *secv, sec_str sec)
{

  secv->shift = NULL;
  secv->circ = NULL;

  secv->shift = (double *) malloc(sec.lagidx * sizeof(double));
  if(secv->shift == NULL)
    fprintf(stderr, "ERROR: Shift Autocorrelation Data Memory allocation did not complete successfully!\n");    

  secv->circ = (double *) malloc(sec.lagidx * sizeof(double));
  if(secv->circ == NULL)
    fprintf(stderr, "ERROR: Circular Autocorrelation Data Memory allocation did not complete successfully!\n");    
}

void malloc_mcmc_vectors(mcmc_int_v *mclocv, mcmc_str mcin)
{
  mclocv->proposed = NULL;
  mclocv->current = NULL;

  mclocv->proposed = (double *) malloc(mcin.ddata * sizeof(double));
  if(mclocv->proposed == NULL)
    fprintf(stderr, "ERROR: Proposed Samples Memory allocation did not complete successfully!\n");
  
  mclocv->current = (double *) malloc(mcin.ddata * sizeof(double));  
  if(mclocv->current == NULL)
    fprintf(stderr, "ERROR: Current Samples Memory allocation did not complete successfully!\n");


  if(mcin.impl == MP)
  {
    mclocv->proposedf = NULL;
    mclocv->currentf = NULL;

    mclocv->proposedf = (float *) malloc(mcin.ddata * sizeof(float));
    if(mclocv->proposedf == NULL)
      fprintf(stderr, "ERROR: Single Precision Proposed Samples Memory allocation did not complete successfully!\n");
    
    mclocv->currentf = (float *) malloc(mcin.ddata * sizeof(float));  
    if(mclocv->currentf == NULL)
      fprintf(stderr, "ERROR: Single Precision Current Samples Memory allocation did not complete successfully!\n");
  }
}

void malloc_autocorrelation_vectors_sp(sec_v_str *secv, sec_str sec)
{
  secv->circf = NULL;

  secv->circf = (float *) malloc(sec.lagidx * sizeof(float));
  if(secv->circf == NULL)
    fprintf(stderr, "ERROR: Circular Autocorrelation Data Memory allocation did not complete successfully!\n");    
}

void malloc_mcmc_vectors_sp(mcmc_int_v *mclocv, mcmc_str mcin)
{
  mclocv->proposedf = NULL;
  mclocv->currentf = NULL;

  mclocv->proposedf = (float *) malloc(mcin.ddata * sizeof(float));
  if(mclocv->proposedf == NULL)
    fprintf(stderr, "ERROR: Proposedf Samples Memory allocation did not complete successfully!\n");
  
  mclocv->currentf = (float *) malloc(mcin.ddata * sizeof(float));  
  if(mclocv->currentf == NULL)
    fprintf(stderr, "ERROR: Currentf Samples Memory allocation did not complete successfully!\n");  
}

void free_data_vectors_sp(data_str data, mcmc_str mcin)
{
  free(data.dataf);
  free(data.mvoutf);
}

void free_data_vectors(data_str data, mcmc_str mcin)
{
  if(mcin.impl == SP)
  {
    free(data.dataf);
    free(data.mvoutf);
  }else{
    free(data.data);

    if(mcin.impl == MP)
    {
      free(data.dataf);  
    }else{
      free(data.mvout);
    }
  }

}

void free_sample_vectors(mcmc_v_str mcdata)
{
  free(mcdata.samples);
  free(mcdata.burn); 
  free(mcdata.nsamples);
  free(mcdata.nburn);  
  free(mcdata.sample_means); 
}

void free_autocorrelation_vectors(sec_v_str secv)
{
  free(secv.shift);  
  free(secv.circ);
}

void free_mcmc_vectors(mcmc_int_v mclocv, mcmc_str mcin)
{
  free(mclocv.proposed);
  free(mclocv.current);
  if(mcin.impl == MP)
  {
    free(mclocv.proposedf);
    free(mclocv.currentf);    
  }
}

void free_sample_vectors_sp(mcmc_v_str mcdata)
{
  free(mcdata.samplesf);
  free(mcdata.burnf); 
  free(mcdata.nsamplesf);
  free(mcdata.nburnf);  
  free(mcdata.sample_meansf); 
}

void free_autocorrelation_vectors_sp(sec_v_str secv)
{ 
  free(secv.circf);
}

void free_mcmc_vectors_sp(mcmc_int_v mclocv, mcmc_str mcin)
{
  free(mclocv.proposedf);
  free(mclocv.currentf);
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

void init_rng_seed(gsl_rng **r, unsigned long int seed)
{
  const gsl_rng_type * T;

  gsl_rng_default_seed = seed;
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

#endif // __ALLOC_UTIL_C__