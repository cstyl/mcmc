#ifndef __IO_HEAD_H__
#define __IO_HEAD_H__

#include "resources.h"
#include "alloc_util.h"
#include "data_util.h"
#include "processing_util.h"

#if defined (__cplusplus)
extern "C" {
#endif

void read_inputs(int an, char *av[], mcmc_str *mcin, sec_str *sec);

void read_inputs_gpu(int an, char *av[], mcmc_str *mcin, 
                        sec_str *sec, gpu_v_str *gpu
                    );
void write_bandwidth_test_out(out_str res);
void write_perf_out_csv(char *rootdir, out_str res, bool first);

void print_normalised_sample_means(mcmc_v_str mcdata, mcmc_str mcin);
void write_csv_outputs(char *rootdir, mcmc_v_str mcdata, mcmc_str mcin, 
                    sec_str sec, sec_v_str secv);

#if defined (__cplusplus)
}
#endif

#endif // __IO_HEAD_H__