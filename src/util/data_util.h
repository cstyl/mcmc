#ifndef __DATA_UTIL_H__
#define __DATA_UTIL_H__

#include "resources.h"
#include "processing_util.h"
#include "csvparser.h"

void read_data(data_str data, mcmc_str mcin, sec_str sec);

void write_data(char *filename, double *data, int sz, int dim);
void write_autocorr(char *filename, double *autocorrelation, sec_str sec);

void output_norm_files(char* dir, mcmc_v_str mcdata, mcmc_str mcin);
void output_files(char* dir, mcmc_v_str mcdata, mcmc_str mcin);
void output_autocorrelation_files(char* dir, sec_v_str secv, sec_str sec);

void output_means(char* dir, mcmc_v_str mcdata, mcmc_str mcin);

#endif  //__DATA_UTIL_H__