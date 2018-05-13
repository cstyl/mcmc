#ifndef __IO_H__
#define __IO_H__

#include "resources.h"
#include "alloc_util.h"
#include "data_util.h"
#include "processing_util.h"

void read_inputs(int an, char *av[], mcmc_str *mcin, sec_str *sec);
void write_outputs(char *rootdir, mcmc_v_str mcdata, 
                    mcmc_str mcin, sec_str sec, 
                    out_str results);
#endif // __IO_H__