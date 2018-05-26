#ifndef __MP_HOST_C__
#define __MP_HOST_C__

#include "mp_host.h"


int main(int argc, char * argv[])
{
  data_str data;

  mcmc_str mcin;
  mcmc_tune_str mct;
  mcmc_v_str mcdata;

  sec_str sec;
  sec_v_str secv;
  
  gpu_v_str gpu;
  out_str res;

  gsl_rng *r = NULL;
  
  char indir[50], outdir[50];
  clock_t start, stop; 

  read_inputs_gpu(argc, argv, &mcin, &sec, &gpu);
  mcin.impl = MP;

  if(sec.fdata == 1){
    strcpy(indir, "data/host/synthetic.csv");
    strcpy(outdir, "out/host/synthetic/mp_");
  }else if(sec.fdata == 2){
    strcpy(indir, "data/host/mnist.csv");
    strcpy(outdir, "out/host/mnist/mp_");
  }

  malloc_data_vectors(&data, mcin);
  malloc_sample_vectors(&mcdata, mcin);
  malloc_autocorrelation_vectors(&secv, sec);

  init_rng(&r);

  read_data(indir, RowMajor, data, mcin);
  
  mct.rwsd = 2.38 / sqrt(mcin.ddata);
  
  int i;
  for(i=0; i<mcin.ddata; i++){ mcdata.burn[i] = 0; } 

  start  = clock();
  mp_sampler(data, r, mcin, &mct, mcdata, gpu, &res);
  stop = clock() - start;

  res.samplerTime = stop * 1000 / CLOCKS_PER_SEC;  // sampler time in ms
  res.ess = get_ess(mcdata.samples, mcin.Ns, mcin.ddata, sec.lagidx, secv.circ);
  
  write_bandwidth_test_out(res);

  calculate_normalised_sample_means(mcdata, mcin);
  print_normalised_sample_means(mcdata, mcin);
  
  write_csv_outputs(outdir, mcdata, mcin, sec, secv);

  free_autocorrelation_vectors(secv);
  free_data_vectors(data, mcin);
  free_sample_vectors(mcdata);
  free_rng(r);

  return 0;
}

#endif // __MP_HOST_C__