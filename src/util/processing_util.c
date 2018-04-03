#include "cpu_host.h"
#include "processing_util.h"

void normalise_samples(int data_dim, int samples, 
                        int burn_samples, double bias_mu)
{
  int samp_idx, dim_idx, idx;
  // normalising burn samples
  for(samp_idx=0; samp_idx<burn_samples; samp_idx++){
    for(dim_idx=0; dim_idx<data_dim; dim_idx++){
      idx = samp_idx*data_dim+dim_idx;
      norm_burn_matrix[idx] = burn_matrix[idx] / bias_mu;
    }
  }
  // normalising parameter samples
  for(samp_idx=0; samp_idx<samples; samp_idx++){
    for(dim_idx=0; dim_idx<data_dim; dim_idx++){
      idx = samp_idx*data_dim+dim_idx;
      norm_sample_matrix[idx] = sample_matrix[idx] / bias_mu;
    }
  }
}

double get_bias_mean(int data_dim, int samples)
{
  int samp_idx;
  double mu = 0;
 
  for(samp_idx=0; samp_idx<samples; samp_idx++){
    mu += sample_matrix[samp_idx*data_dim];  // get bias mean
  }

  return mu/samples;
}

double shift_autocorrelation(double *matrix, int data_dim, 
                              int samples, int lag_idx)
{ 
  double *mean = (double *) malloc(data_dim * sizeof(double));;
  double *variance = (double *) malloc(data_dim * sizeof(double));
  double *lag = (double *) malloc(data_dim * sizeof(double));
  
  double autocorrelation_sum = 0;
  
  int k;
  int samp_idx,dim_idx;

  for(k=0; k<=lag_idx; k++){  // test for various lagk to find where samples become independent
    get_mean(data_dim, samples-k, matrix, mean);
    get_variance(data_dim, samples-k, matrix, variance, mean);

    memset(lag, 0, data_dim * sizeof(double));

    for(samp_idx=0; samp_idx<samples-k; samp_idx++){
      for(dim_idx=0; dim_idx<data_dim; dim_idx++){
        lag[dim_idx] += (matrix[samp_idx*data_dim+dim_idx] - mean[dim_idx]) 
                        * (matrix[(samp_idx+k)*data_dim+dim_idx] - mean[dim_idx]);
      }
    }
    double prod = 1;
    for(dim_idx=0; dim_idx<data_dim; dim_idx++){
      prod *= lag[dim_idx] / variance[dim_idx]; // assuming iid then multiply each dimension
    }  
    autocorrelation_shift[k] = prod;
    autocorrelation_sum += autocorrelation_shift[k];
  }

  autocorrelation_sum -= autocorrelation_shift[0]; // remove lag0 contribution

  free(mean);
  free(variance);
  free(lag);
  return autocorrelation_sum;
}

double circular_autocorrelation(double *matrix, int data_dim, 
                                int samples, int lag_idx)
{
  double *mean = (double *) malloc(data_dim * sizeof(double));;
  double *variance = (double *) malloc(data_dim * sizeof(double));
  double *lag = (double *) malloc(data_dim * sizeof(double));

  double autocorrelation_sum = 0;
  
  int samp_idx, dim_idx;
  int k,shift_idx;
  bool wrap;
 
  get_mean(data_dim, samples, matrix, mean);
  get_variance(data_dim, samples, matrix, variance, mean);

  for(k=0; k<lag_idx; k++){
    memset(lag, 0, data_dim * sizeof(double));

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
        lag[dim_idx] += (matrix[samp_idx * data_dim + dim_idx] - mean[dim_idx]) 
                        * (matrix[shift_idx * data_dim + dim_idx] - mean[dim_idx]);
      }
    }

    double prod = 1;
    for(dim_idx = 0; dim_idx < data_dim; dim_idx++){
      prod *= lag[dim_idx] / variance[dim_idx]; // assuming iid then multiply each dimension
    }  
    autocorrelation_circ[k] = prod;
    autocorrelation_sum += autocorrelation_circ[k];       
  }
  
  autocorrelation_sum -= autocorrelation_circ[0]; // remove lag0 contribution
  
  free(mean);
  free(variance);
  free(lag);

  return autocorrelation_sum;
}

void get_mean(int data_dim, int samples, 
              double *data, double *mu)
{
  int samp_idx, dim_idx;
  memset(mu, 0, data_dim * sizeof(double));

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
  int samp_idx, dim_idx;
  memset(var, 0, data_dim * sizeof(double)); 

  for(samp_idx = 0; samp_idx < samples; samp_idx++){
    for(dim_idx = 0; dim_idx < data_dim; dim_idx++){
      var[dim_idx] += pow(data[samp_idx * data_dim + dim_idx] 
                        - mu[dim_idx],2);
    }
  }
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