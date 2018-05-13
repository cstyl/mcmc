// #include "cpu_host.h"
// #include "gpu_host.h"
#include "processing_util.h"

void normalise_samples(double *raw_in, double *norm_out,
                      int data_dim, int sz, double bias_mu)
{
  int samp_idx, dim_idx, idx;
  // double bias_mu = get_bias_mean(raw_in, data_dim, sz);

  for(samp_idx=0; samp_idx<sz; samp_idx++){
    for(dim_idx=0; dim_idx<data_dim; dim_idx++){
      idx = samp_idx*data_dim+dim_idx;
      norm_out[idx] = raw_in[idx] / bias_mu;
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

void map_gpu_data(data_str data, mcmc_str mcin)
{
  int i,j;

  if(mcin.ddata == mcin.dmap){
    for(i=0; i<mcin.Nd; i++){
      for(j=0; j<mcin.ddata; j++){
        data.gpudata[i * mcin.dmap + j] = data.data[i * mcin.dmap + j];
      }
    }    
  }else{ // pad with zeros
    for(i=0; i<mcin.Nd; i++){
      for(j=0; j<mcin.ddata; j++){
        data.gpudata[i * mcin.dmap + j] = data.data[i * mcin.dmap + j];
      }
      for(j=mcin.ddata; j<mcin.dmap; j++)
      {
        data.gpudata[i * mcin.dmap + j] = 0;
      }
    }
  }
}

// make dimensionality of data power of 2
int map_dimensions(int d_data){
  int ctr = 0;

  while(d_data!=0){
    d_data = d_data/2;
    ctr++;
  }

  return pow(2,ctr-1);
}

// void map_gpu_data()
// {
//   int warp_sz = 32;
//   int total_threads; // multiple of 32

//   total_threads = ceil(Nd * gpu_d_data / warp_sz) * warp_sz; // make number of threads multiple of 32
//   block_number = (total_threads + threads_per_block - 1) / threads_per_block;  // number of blocks

//   int i,j;
//   // copy data and zero-pad new dimension (if any)
//   for(i=0; i<Nd; i++){
//     for(j=0; j<gpu_d_data; j++){
//       gpu_x[i*gpu_d_data+j] = x[i*gpu_d_data+j];
//       if(j>d_data-1){
//         // zero pad
//         gpu_x[i*gpu_d_data+j] = 0;
//       }
//     }
//   }
// }