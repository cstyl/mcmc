/*
 * Implementation of MCMC Metropolis-Hastings Algorithm on CPU
 * Aim: Obtain a sequence of RANDOM samples from a probability distribution
 * from which direct sampling is difficult
 * The sequence can be used to approximate the distribution (eg histogram) 
 * or compute an integral (eg expected value)
 */

#include "cpu_host.h"

int main(int argc, char * argv[])
{
  data_str data;

  mcmc_str mcin;
  mcmc_tune_str mct;
  mcmc_v_str mcdata;

  sec_str sec;
  out_str results;

  gsl_rng *r = NULL;
  
  char rootdir[50];

  read_inputs(argc, argv, &mcin, &sec);

  init_rng(&r);
  malloc_data_vectors_cpu(&data, mcin);
  malloc_sample_vectors(&mcdata, mcin);
  
  read_data(data, mcin, sec);
  
  mct.rwsd = 2.38 / sqrt(mcin.ddata);

  int i;
  for(i=0; i<mcin.ddata; i++) mcdata.burn[i] = 0;

  // Metropolis_Hastings_cpu(data, r, mcin, mct, mcdata, &results);
  cpu_sampler(data, r, mcin, mct, mcdata, &results);


  if(sec.fdata == 1){
    strcpy(rootdir, "out/cpu/synthetic/");
  }else if(sec.fdata == 2){
    strcpy(rootdir, "out/cpu/mnist/");
  }
  write_outputs(rootdir, mcdata, mcin, sec, results);

  free_data_vectors_cpu(data);
  free_sample_vectors(mcdata);
  free_rng(r);

  return 0;
}