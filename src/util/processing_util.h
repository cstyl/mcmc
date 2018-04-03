#ifndef __PROCESSING_UTIL_H__
#define __PROCESSING_UTIL_H__

void normalise_samples(int data_dim, int samples, 
                        int burn_samples, double bias_mu);
double get_bias_mean(int data_dim, int samples);

double shift_autocorrelation(double *matrix, int data_dim, 
                              int samples, int lag_idx);
double circular_autocorrelation(double *matrix, int data_dim, 
                                int samples, int lag_idx);
void get_mean(int data_dim, int samples, 
              double *data, double *mu);
void get_variance(int data_dim, int samples,
                  double *data, double *var, double *mu);

#endif  //__PROCESSING_UTIL_H__