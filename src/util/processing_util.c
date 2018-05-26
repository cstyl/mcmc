#ifndef __PROCESSING_UTIL_C__
#define __PROCESSING_UTIL_C__

#include "processing_util.h"

#define MIN(x,y) ((x < y) ? x : y)

void getSizes_mp(sz_str *sz, mcmc_str mcin, gpu_v_str gpu)
{
  sz->samples = mcin.ddata * sizeof(double);
  sz->samplesf = mcin.ddata * sizeof(float);

  sz->data = mcin.ddata * mcin.Nd * sizeof(double);
  sz->dataf = mcin.ddata * mcin.Nd * sizeof(float);

  sz->cuLhood = mcin.Nd * sizeof(double);
  sz->cuLhoodf = mcin.Nd * sizeof(float);

  sz->lhood = gpu.blocks * sizeof(double);

  sz->zlabels = mcin.Nd * sizeof(int8_t);
  sz->zidx = mcin.Nd * sizeof(int);

  sz->brightLhood = mcin.Nd * sizeof(double);   
  sz->darkLhood = mcin.Nd * sizeof(float);       

  sz->resample = 2 * mcin.Nd * sizeof(double);   // requires double the memory

  sz->dLhood = mcin.Nd * sizeof(double);
}

// int nextPow2(int d_data){
//   int ctr = 0;

//   while(d_data!=0){
//     d_data = d_data/2;
//     ctr++;
//   }

//   return pow(2,ctr);
// }
int nextPow2(int d_data)
{
    --d_data;
    d_data |= d_data >> 1;
    d_data |= d_data >> 2;
    d_data |= d_data >> 4;
    d_data |= d_data >> 8;
    d_data |= d_data >> 16;
    return ++d_data;
}

void getBlocksAndThreads(int kernel, int n, int maxBlocks, int maxThreads, int *blocks, int *threads)
{
    int lthreads, lblocks;

    if(kernel<3)
    {
        // if data less than the # of threads per block make threads per block the next power of 2
        // otherwise use the preset maximum threads
        lthreads = (n < maxThreads) ? nextPow2(n) : maxThreads;  
        lblocks = (n + lthreads - 1) / lthreads;
    }else{
        lthreads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        lblocks = (n + (lthreads * 2 - 1)) / (lthreads * 2);
    }

    if(kernel == 6) lblocks = MIN(maxBlocks, lblocks);

    *threads = lthreads;
    *blocks = lblocks;
}

void normalise_samples(double *raw_in, double *norm_out,
                      int data_dim, int sz)
{
  int samp_idx, dim_idx, idx;
  // double bias_mu = get_bias_mean(raw_in, data_dim, sz);

  for(samp_idx=0; samp_idx<sz; samp_idx++){
    for(dim_idx=0; dim_idx<data_dim; dim_idx++){
      idx = samp_idx*data_dim+dim_idx;
      norm_out[idx] = raw_in[idx] / raw_in[samp_idx*data_dim];
    }
  }
}

double get_bias_mean(double *in_vec, int data_dim, int sample_sz)
{
  int samp_idx;
  double mu = 0;
 
  for(samp_idx=0; samp_idx<sample_sz; samp_idx++){
    mu += in_vec[samp_idx*data_dim];  // get bias mean
  }

  return mu/sample_sz;
}

double get_dim_mean(double *in_vec, int data_dim, int current_dim, int sample_sz)
{
  int samp_idx;
  double mu = 0;
 
  for(samp_idx=0; samp_idx<sample_sz; samp_idx++){
    mu += in_vec[samp_idx*data_dim + current_dim];  // get bias mean
  }

  return mu/sample_sz;  
}

double shift_autocorrelation(double *out_v, double *in_m, int data_dim, 
                              int samples, int lag_idx)
{ 
  double *mean = (double *) malloc(data_dim * sizeof(double));;
  double *variance = (double *) malloc(data_dim * sizeof(double));
  double *autocorrelation = (double *) malloc(data_dim * sizeof(double));
  
  double autocorrelation_sum = 0;
  
  int i,k;
  int samp_idx,dim_idx;
  
  get_mean(data_dim, samples, in_m, mean);
  get_variance(data_dim, samples, in_m, variance, mean);

  for(k=0; k<=lag_idx; k++){  // test for various lagk to find where samples become independent

    for(i=0; i<data_dim; i++)
      autocorrelation[i] = 0;

    for(samp_idx=0; samp_idx<samples-k; samp_idx++){
      for(dim_idx=0; dim_idx<data_dim; dim_idx++){
        autocorrelation[dim_idx] += (in_m[samp_idx*data_dim+dim_idx] - mean[dim_idx]) 
                        * (in_m[(samp_idx+k)*data_dim+dim_idx] - mean[dim_idx]);
      }
    }
    double prod = 1;
    for(dim_idx=0; dim_idx<data_dim; dim_idx++){
      prod *= autocorrelation[dim_idx] / (variance[dim_idx]*(samples-k)); // assuming iid then multiply each dimension
    }  
    out_v[k] = prod;
    autocorrelation_sum += out_v[k];
  }

  autocorrelation_sum -= out_v[0]; // remove lag0 contribution

  free(mean);
  free(variance);
  free(autocorrelation);
  return autocorrelation_sum;
}

double circular_autocorrelation(double *out_v, double *in_m, int data_dim, 
                                int samples, int lag_idx)
{
  double *mean = (double *) malloc(data_dim * sizeof(double));
  double *variance = (double *) malloc(data_dim * sizeof(double));
  double *autocorrelation = (double *) malloc(data_dim * sizeof(double));

  double autocorrelation_sum = 0;
  
  int samp_idx, dim_idx;
  int i, k, shift_idx;
  bool wrap;
 
  get_mean(data_dim, samples, in_m, mean);
  get_variance(data_dim, samples, in_m, variance, mean);

  for(k=0; k<lag_idx; k++){
    for(i=0; i<data_dim; i++)
      autocorrelation[i] = 0;

    wrap = false;
    for(samp_idx=0; samp_idx<samples; samp_idx++){
      if(!wrap){
        shift_idx = samp_idx+k;
      }else{
        shift_idx++;
      }

      if(shift_idx>samples){
        shift_idx = 0;  // wrap around index
        wrap = true;
      }

      for(dim_idx=0; dim_idx<data_dim; dim_idx++){
        autocorrelation[dim_idx] += (in_m[samp_idx * data_dim + dim_idx] - mean[dim_idx]) 
                        * (in_m[shift_idx * data_dim + dim_idx] - mean[dim_idx]);
      }
    }

    double prod = 1;
    for(dim_idx = 0; dim_idx < data_dim; dim_idx++){
      prod *= autocorrelation[dim_idx] / (variance[dim_idx]*(samples-k)); // assuming iid then multiply each dimension
    }  
    out_v[k] = prod;
    autocorrelation_sum += out_v[k];       
  }
  
  autocorrelation_sum -= out_v[0]; // remove lag0 contribution
  
  free(mean);
  free(variance);
  free(autocorrelation);

  return autocorrelation_sum;
}

void get_mean(int data_dim, int samples, 
              double *data, double *mu)
{
  int samp_idx, dim_idx, i;
  for(i=0; i<data_dim; i++)
    mu[i] = 0;

  // get mean vector
  for(samp_idx = 0; samp_idx < samples; samp_idx++){
    for(dim_idx = 0; dim_idx < data_dim; dim_idx++){
      mu[dim_idx] += data[samp_idx * data_dim + dim_idx];
    }
  }

  for(dim_idx=0; dim_idx<data_dim; dim_idx++){
    mu[dim_idx] /= samples;
  }
}

void get_variance(int data_dim, int samples,
                  double *data, double *var, double *mu)
{
  int samp_idx, dim_idx, i;
  for(i=0; i<data_dim; i++)
    var[i] = 0;

  for(samp_idx = 0; samp_idx < samples; samp_idx++){
    for(dim_idx = 0; dim_idx < data_dim; dim_idx++){
      var[dim_idx] += pow(data[samp_idx * data_dim + dim_idx] 
                        - mu[dim_idx],2);
    }
  }

  for(dim_idx=0; dim_idx<data_dim; dim_idx++){
    var[dim_idx] /= samples;
  }
}

double get_ess(double *samples, int sample_sz, int sample_dim, int max_lag, double *autocorrelation_v)
{
  double autocorrelation_sum, ess;

  autocorrelation_sum = circular_autocorrelation(autocorrelation_v, samples, sample_dim, sample_sz, max_lag);

  ess =  sample_sz / (1 + 2 * autocorrelation_sum);

  return ess;
}

void calculate_normalised_sample_means(mcmc_v_str mcdata, mcmc_str mcin)
{
  // normalise samples for bias = 1
  normalise_samples(mcdata.burn, mcdata.nburn, mcin.ddata, mcin.burnin); 
  normalise_samples(mcdata.samples, mcdata.nsamples, mcin.ddata, mcin.Ns);
  //get normalised mean for each dimension
  int current_idx;
  for(current_idx = 0; current_idx < mcin.ddata; current_idx++){
    mcdata.sample_means[current_idx] = get_dim_mean(mcdata.nsamples, mcin.ddata, current_idx, mcin.Ns);
  }
}

#endif  // __PROCESSING_UTIL_C__