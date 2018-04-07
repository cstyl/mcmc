#ifndef __DATA_UTIL_H__
#define __DATA_UTIL_H__

#include "resources.h"
#include "processing_util.h"
#include "csvparser.h"

void read_data(double *data, double *labels, 
                int data_dim, int datasz, int test);
void write_data(char *filename, double *data, 
                int data_dim, int data_size);
void write_autocorr(char *filename, double *autocorrelation, int lag_idx);
void output_norm_files(char* dir, double *norm_samp_m, double *norm_burned_m,
                        int data_dim, int samples, int burn_samples);
void output_files(char* dir, double *samp_m, double *burned_m,
                  int data_dim, int samples, int burn_samples);
void output_autocorrelation_files(char* dir, double *shift_v, double *circ_v,
                                  int auto_case, int lag);
#endif  //__DATA_UTIL_H__