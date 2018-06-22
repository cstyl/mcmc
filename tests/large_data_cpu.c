/*
 * Implementation of MCMC Metropolis-Hastings Algorithm on CPU
 * Aim: Obtain a sequence of RANDOM samples from a probability distribution
 * from which direct sampling is difficult
 * The sequence can be used to approximate the distribution (eg histogram) 
 * or compute an integral (eg expected value)
 */
#ifndef __LARGE_DATA_CPU_C__
#define __LARGE_DATA_CPU_C__

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
  FILE *fp;
  gsl_rng *r = NULL;
  
  char indir[50], outdir[50];
  clock_t start, stop;

  init_rng(&r);

  read_inputs(argc, argv, &mcin, &sec);
  mcin.impl = CPU;

  strcpy(indir, "data/large_data_cpu/synthetic.csv");  
  if(sec.fdata == 3){
    strcpy(outdir, "out/large_data_cpu/cpu_500.csv");
  }else if(sec.fdata==4){
    strcpy(outdir, "out/large_data_cpu/cpu_200.csv");
  }else if(sec.fdata == 5){
    strcpy(outdir, "out/large_data_cpu/cpu_100.csv");
  }else if(sec.fdata == 6){
    strcpy(outdir, "out/large_data_cpu/cpu_50.csv");
  }else if(sec.fdata == 7){
    strcpy(outdir, "out/large_data_cpu/cpu_20.csv");
  }else if(sec.fdata == 8){
    strcpy(outdir, "out/large_data_cpu/cpu_10.csv");
  }else if(sec.fdata == 9){
    strcpy(outdir, "out/large_data_cpu/cpu_3.csv");
  }

  malloc_data_vectors(&data, mcin);
  malloc_sample_vectors(&mcdata, mcin);
  malloc_autocorrelation_vectors(&secv, sec);
  
  read_data(indir, ColMajor, data, mcin);
  int Nd[] = {10, 20, 50, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 300000, 400000, 500000};

  int i, ndIdx;
  fp=fopen(outdir,"w+");
  fprintf(fp, "Dim, Nd, sampler_t(ms), metrop_t(ms), burn_t(ms)\n"); 

  for(ndIdx=0; ndIdx<16; ndIdx++)
  {
    mcin.Nd = Nd[ndIdx];
    // if(ndIdx<10)
    // {
    //   mcin.Ns = 10000;
    //   mcin.burnin = 2000;
    // }else{
    //   mcin.Ns = 1000;
    //   mcin.burnin = 200;      
    // }
    mct.rwsd = 2.38 / sqrt(mcin.ddata);
    for(i=0; i<mcin.ddata; i++) mcdata.burn[i] = 0;

    // tune_ess_cpu(data, r, mcin, &mct, mcdata.burn, 1000); 

    printf("Idx: %d...Starting run for Nd:%d, dim:%d\n", ndIdx, mcin.Nd, mcin.ddata);
    start  = clock();
    cpu_sampler(data, r, mcin, mct, mcdata, &res);
    stop = clock() - start;
    printf("Idx: %d... Run for Nd:%d, dim:%d Completed..\n", ndIdx, mcin.Nd, mcin.ddata);
    res.samplerTime = stop * 1000 / CLOCKS_PER_SEC;  // sampler time in ms
    // if(ndIdx<10)
    //   fprintf(fp, "%d, %d, %.32f, %.32f, %.32f\n", mcin.ddata, mcin.Nd, res.samplerTime, res.mcmcTime, res.burnTime);
    // else
    //   fprintf(fp, "%d, %d, %.32f, %.32f, %.32f\n", mcin.ddata, mcin.Nd, res.samplerTime*10, res.mcmcTime*10, res.burnTime*10);
    fprintf(fp, "%d, %d, %.32f, %.32f, %.32f\n", mcin.ddata, mcin.Nd, res.samplerTime, res.mcmcTime, res.burnTime);
  }

  fclose(fp);

  free_autocorrelation_vectors(secv);
  free_data_vectors(data, mcin);
  free_sample_vectors(mcdata);
  free_rng(r);

  return 0;
}

#endif // __LARGE_DATA_CPU_C__