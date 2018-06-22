#ifndef __DATA_UTIL_H__
#define __DATA_UTIL_H__

#include "resources.h"
#include "processing_util.h"
#include "csvparser.h"

#if defined (__cplusplus)
extern "C" {
#endif

void read_data(char* dir, int store, data_str data, mcmc_str mcin);
void read_data_sp(char* dir, int store, data_str data, mcmc_str mcin);

void write_data(char *filename, double *data, int sz, int dim);
void write_autocorr(char *filename, double *autocorrelation, sec_str sec);

void write_performance_gpu(char *filename, out_str res);
void write_performance_data(char *filename, out_str res, bool first);

void output_norm_files(char* dir, mcmc_v_str mcdata, mcmc_str mcin);
void output_files(char* dir, mcmc_v_str mcdata, mcmc_str mcin);
void output_autocorrelation_files(char* dir, sec_v_str secv, sec_str sec);

void output_means(char* dir, mcmc_v_str mcdata, mcmc_str mcin);


void output_norm_files_sp(char* dir, mcmc_v_str mcdata, mcmc_str mcin);
void output_means_sp(char* dir, mcmc_v_str mcdata, mcmc_str mcin);
void output_files_sp(char* dir, mcmc_v_str mcdata, mcmc_str mcin);
void output_autocorrelation_files_sp(char* dir, sec_v_str secv, sec_str sec);
void write_data_sp(char *filename, float *data, int sz, int dim);
void write_autocorr_sp(char *filename, float *autocorrelation, sec_str sec);
#if defined (__cplusplus)
}
#endif

#endif  //__DATA_UTIL_H__