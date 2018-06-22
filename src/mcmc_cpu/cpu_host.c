/*
 * Implementation of MCMC Metropolis-Hastings Algorithm on CPU
 * Aim: Obtain a sequence of RANDOM samples from a probability distribution
 * from which direct sampling is difficult
 * The sequence can be used to approximate the distribution (eg histogram) 
 * or compute an integral (eg expected value)
 */
#ifndef __CPU_HOST_C__
#define __CPU_HOST_C__

#include "cpu_host.h"

int main(int argc, char * argv[])
{
  data_str data;

  mcmc_str mcin;
  mcmc_tune_str mct;
  mcmc_v_str mcdata;

  sec_str sec;
  sec_v_str secv;
  out_str res;

  gsl_rng *r = NULL;
  
  char indir[50], outdir[50];
  clock_t start, stop;

  init_rng(&r);

  read_inputs(argc, argv, &mcin, &sec);
  mcin.impl = CPU;
  
  if(sec.fdata == 1){
    strcpy(indir, "data/host/synthetic.csv");
    strcpy(outdir, "out/host/synthetic/cpu_");
  }else if(sec.fdata == 2){
    strcpy(indir, "data/host/mnist.csv");
    strcpy(outdir, "out/host/mnist/cpu_");
  }

  malloc_data_vectors(&data, mcin);
  malloc_sample_vectors(&mcdata, mcin);
  malloc_autocorrelation_vectors(&secv, sec);
  
  read_data(indir, ColMajor, data, mcin);

  mct.rwsd = 2.38 / sqrt(mcin.ddata);

  int i;
  for(i=0; i<mcin.ddata; i++) mcdata.burn[i] = 0;
  
  clock_t startTune, stopTune;
  startTune = clock();
  if(mcin.tune == 1)
    tune_target_a_cpu_v2(data, r, mcin, &mct, mcdata.burn, 0.25, 40);
  else if(mcin.tune == 2)  
    tune_ess_cpu(data, r, mcin, &mct, mcdata.burn, 5000);    
  stopTune = clock() - startTune;
  res.tuneTime = stopTune * 1000 / CLOCKS_PER_SEC;   // tuning time in ms


  start  = clock();
  cpu_sampler(data, r, mcin, mct, mcdata, &res);
  stop = clock() - start;

  res.samplerTime = stop * 1000 / CLOCKS_PER_SEC;  // sampler time in ms
  res.ess = get_ess(mcdata.samples, mcin.Ns, mcin.ddata, sec.lagidx, secv.circ);

  write_test_out(res);

  calculate_normalised_sample_means(mcdata, mcin);
  print_normalised_sample_means(mcdata, mcin);
  
  write_csv_outputs(outdir, mcdata, mcin, sec, secv);

  free_autocorrelation_vectors(secv);
  free_data_vectors(data, mcin);
  free_sample_vectors(mcdata);
  free_rng(r);

  return 0;
}

#endif // __CPU_HOST_C__