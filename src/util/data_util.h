#ifndef __DATA_UTIL_H__
#define __DATA_UTIL_H__

void read_data(int data_dim, int data, int test);
void write_data(char *filename, double *data, 
                int data_dim, int data_size);
void write_autocorr(char *filename, double *autocorrelation, int lag_idx);
void output_norm_files(char* dir, int data_dim, 
                        int samples, int burn_samples);
void output_files(char* dir, int data_dim,
                  int samples, int burn_samples);
void output_autocorrelation_files(char* dir, int auto_case, int lag);
#endif  //__DATA_UTIL_H__