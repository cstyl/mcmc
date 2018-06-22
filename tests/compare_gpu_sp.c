/*
 * Implementation of MCMC Metropolis-Hastings Algorithm on CPU
 * Aim: Obtain a sequence of RANDOM samples from a probability distribution
 * from which direct sampling is difficult
 * The sequence can be used to approximate the distribution (eg histogram) 
 * or compute an integral (eg expected value)
 */
#ifndef __COMPARE_GPU_SP_C__
#define __COMPARE_GPU_SP_C__

#include "gpu_sp_host.h"
#include "gpu_host.h"

int main(int argc, char * argv[])
{
  data_str data;

  mcmc_str mcin;
  mcmc_tune_str mct;
  mcmc_v_str mcdata;

  sec_str sec;
  sec_v_str secv;
  out_str res;
  FILE *fpsp, *fpgpu;
  gsl_rng *r = NULL;
  gpu_v_str gpu;  
  char indir[50], outdirsp[50], outdirgpu[50];
  clock_t start, stop;

  init_rng(&r);

  read_inputs_gpu(argc, argv, &mcin, &sec, &gpu);
  mcin.impl = CPU;

  if(sec.fdata <8){
    strcpy(indir, "data/compare_gpu_sp/synthetic.csv");  
    if(sec.fdata==3){
      strcpy(outdirsp, "out/compare_gpu_sp/sp_200.csv");
      strcpy(outdirgpu, "out/compare_gpu_sp/gpu_200.csv");
    }else if(sec.fdata == 4){
      strcpy(outdirsp, "out/compare_gpu_sp/sp_100.csv");
      strcpy(outdirgpu, "out/compare_gpu_sp/gpu_100.csv");
    }else if(sec.fdata == 5){
      strcpy(outdirsp, "out/compare_gpu_sp/sp_50.csv");
      strcpy(outdirgpu, "out/compare_gpu_sp/gpu_50.csv");
    }else if(sec.fdata == 6){
      strcpy(outdirsp, "out/compare_gpu_sp/sp_20.csv");
      strcpy(outdirgpu, "out/compare_gpu_sp/gpu_20.csv");
    }else if(sec.fdata == 7){
      strcpy(outdirsp, "out/compare_gpu_sp/sp_10.csv");
      strcpy(outdirgpu, "out/compare_gpu_sp/gpu_10.csv");
    }
  }else{
    strcpy(indir, "data/compare_gpu_sp_mnist/mnist.csv");  
    if(sec.fdata==8){
      strcpy(outdirsp, "out/compare_gpu_sp_mnist/sp_100.csv");
      strcpy(outdirgpu, "out/compare_gpu_sp_mnist/gpu_100.csv");
    }else if(sec.fdata == 9){
      strcpy(outdirsp, "out/compare_gpu_sp_mnist/sp_70.csv");
      strcpy(outdirgpu, "out/compare_gpu_sp_mnist/gpu_70.csv");
    }else if(sec.fdata == 10){
      strcpy(outdirsp, "out/compare_gpu_sp_mnist/sp_30.csv");
      strcpy(outdirgpu, "out/compare_gpu_sp_mnist/gpu_30.csv");
    }else if(sec.fdata == 11){
      strcpy(outdirsp, "out/compare_gpu_sp_mnist/sp_12.csv");
      strcpy(outdirgpu, "out/compare_gpu_sp_mnist/gpu_12.csv");
    }else if(sec.fdata == 12){
      strcpy(outdirsp, "out/compare_gpu_sp_mnist/sp_5.csv");
      strcpy(outdirgpu, "out/compare_gpu_sp_mnist/gpu_5.csv");
    }    
  }

  malloc_data_vectors(&data, mcin);
  malloc_sample_vectors(&mcdata, mcin);
  malloc_autocorrelation_vectors(&secv, sec);
  malloc_data_vectors_sp(&data, mcin);
  malloc_sample_vectors_sp(&mcdata, mcin);
  malloc_autocorrelation_vectors_sp(&secv, sec);
  
  read_data(indir, ColMajor, data, mcin);
  read_data_sp(indir,ColMajor,data,mcin);

  // int Nd[] = {500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 300000, 400000, 500000};

  int i;
  fpsp=fopen(outdirsp,"a");
  // fprintf(fpsp, "Dim, Nd, sampler_t(ms), metrop_t(ms), burn_t(ms), ess, ess/t(s)\n"); 
  fpgpu=fopen(outdirgpu,"a");
  // fprintf(fpgpu, "Dim, Nd, sampler_t(ms), metrop_t(ms), burn_t(ms), ess, ess/t(s)\n"); 
 
  mct.rwsdf = 2.38 / (float) (sqrt(mcin.ddata));
  for(i=0; i<mcin.ddata; i++) mcdata.burnf[i] = 0;

  if(mcin.tune == 1)
    tune_target_a_sp_v2(data, r, mcin, &mct, gpu, mcdata.burnf, 0.25, 100);
  else if(mcin.tune == 2)  
    tune_ess_sp(data, r, mcin, &mct, gpu, mcdata.burnf, 5000); 

  printf("Starting run for Nd:%d, dim:%d\n", mcin.Nd, mcin.ddata);
  start  = clock();
  sp_sampler(data, r, mcin, mct, mcdata, gpu, &res);
  stop = clock() - start;
  printf("Run for Nd:%d, dim:%d Completed..\n", mcin.Nd, mcin.ddata);
  res.samplerTime = stop * 1000 / CLOCKS_PER_SEC;  // sampler time in ms
  res.ess = get_ess_sp(mcdata.samplesf, mcin.Ns, mcin.ddata, sec.lagidx, secv.circf);
  fprintf(fpsp, "%d, %d, %.32f, %.32f, %.32f, %.32f, %.32f\n", mcin.ddata, mcin.Nd, res.samplerTime, res.mcmcTime, res.burnTime, res.ess, res.ess/(res.samplerTime*1000));

  mct.rwsd = 2.38 / sqrt(mcin.ddata);
  for(i=0; i<mcin.ddata; i++) mcdata.burn[i] = 0;

  if(mcin.tune == 1)
    tune_target_a_gpu_v2(data, r, mcin, &mct, gpu, mcdata.burn, 0.25, 100);
  else if(mcin.tune == 2)  
    tune_ess_gpu(data, r, mcin, &mct, gpu, mcdata.burn, 5000); 
  printf("Starting run for Nd:%d, dim:%d\n", mcin.Nd, mcin.ddata);
  start  = clock();
  gpu_sampler(data, r, mcin, mct, mcdata, gpu, &res);
  stop = clock() - start;
  printf("Run for Nd:%d, dim:%d Completed..\n", mcin.Nd, mcin.ddata);
  res.samplerTime = stop * 1000 / CLOCKS_PER_SEC;  // sampler time in ms
  res.ess = get_ess(mcdata.samples, mcin.Ns, mcin.ddata, sec.lagidx, secv.circ);
  fprintf(fpgpu, "%d, %d, %.32f, %.32f, %.32f, %.32f, %.32f\n", mcin.ddata, mcin.Nd, res.samplerTime, res.mcmcTime, res.burnTime, res.ess, res.ess/(res.samplerTime*1000));


  fclose(fpsp);
  fclose(fpgpu);

  free_autocorrelation_vectors(secv);
  free_data_vectors(data, mcin);
  free_sample_vectors(mcdata);
  free_autocorrelation_vectors_sp(secv);
  free_data_vectors_sp(data, mcin);
  free_sample_vectors_sp(mcdata);
  free_rng(r);

  return 0;
}

#endif // __COMPARE_GPU_CPU_C__