#ifndef __IO_C__
#define __IO_C__

#include "io.h"

void read_inputs(int an, char *av[], mcmc_str *mcin, sec_str *sec)
{
  int ai = 1;
  while(ai<an)
  {
    if(!strcmp(av[ai],"-d")){ // choose dataset. can be from 1 to 2
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -d.");
        exit(1);        
      }else if((atoi(av[ai+1]) > 2) || (atoi(av[ai+1]) < 1)){
        fprintf(stderr, "Please enter a valid dataset value.");
        exit(1);   
      }
      sec->fdata = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-sz")){ // choose dataset size.
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -sz.");
        exit(1);        
      }
      mcin->Nd = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-dim")){ // choose dataset dimensionality (including bias).
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -dim.");
        exit(1);        
      }
      mcin->ddata = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-samp")){ // choose number of produced samples.
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -samp.");
        exit(1);        
      }
      mcin->Ns = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-burn")){ // choose number of burn samples.
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -burn.");
        exit(1);        
      }
      mcin->burnin = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-lag")){ // choose lag idx
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -lag.");
        exit(1);        
      }else if((atoi(av[ai+1]) < 0)){
        fprintf(stderr, "Please enter a valid value.");
        exit(1);   
      }
      sec->lagidx = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-tune")){ // choose which tuning to perform
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -tune.");
        exit(1);        
      }else if((atoi(av[ai+1]) < 0) || (atoi(av[ai+1]) > 2) ){
        fprintf(stderr, "Please enter a valid value.");
        exit(1);   
      }
      mcin->tune = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-first")){ // choose which tuning to perform
      sec->first = atoi(av[ai+1]);
      ai += 2;
    }
  }
}

void read_inputs_gpu(int an, char *av[], mcmc_str *mcin, sec_str *sec, gpu_v_str *gpu)
{
  int ai = 1;
  while(ai<an)
  {
    if(!strcmp(av[ai],"-d")){ // choose dataset. can be from 1 to 2
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -d.");
        exit(1);        
      }else if((atoi(av[ai+1]) > 2) || (atoi(av[ai+1]) < 1)){
        fprintf(stderr, "Please enter a valid dataset value.");
        exit(1);   
      }
      sec->fdata = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-sz")){ // choose dataset size.
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -sz.");
        exit(1);        
      }
      mcin->Nd = atoi(av[ai+1]);
      gpu->size = atoi(av[ai+1]); 
      ai += 2;
    }else if(!strcmp(av[ai],"-dim")){ // choose dataset dimensionality (including bias).
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -dim.");
        exit(1);        
      }
      mcin->ddata = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-samp")){ // choose number of produced samples.
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -samp.");
        exit(1);        
      }
      mcin->Ns = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-burn")){ // choose number of burn samples.
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -burn.");
        exit(1);        
      }
      mcin->burnin = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-lag")){ // choose lag idx
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -lag.");
        exit(1);        
      }else if((atoi(av[ai+1]) < 0)){
        fprintf(stderr, "Please enter a valid value.");
        exit(1);   
      }
      sec->lagidx = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-tune")){ // choose which tuning to perform
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -tune.");
        exit(1);        
      }else if((atoi(av[ai+1]) < 0) || (atoi(av[ai+1]) > 2) ){
        fprintf(stderr, "Please enter a valid value.");
        exit(1);   
      }
      mcin->tune = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-maxThreads")){ // choose number of threads per block
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -maxThreads.");
        exit(1);        
      }else if(atoi(av[ai+1]) < 0){
        fprintf(stderr, "Please enter a valid value.");
        exit(1);   
      }
      gpu->maxThreads = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-kernel")){ // choose which kernel to perform
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -kernel.");
        exit(1);        
      }else if((atoi(av[ai+1]) < 0) || (atoi(av[ai+1]) > 7)){
        fprintf(stderr, "Please enter a valid value.");
        exit(1);   
      }
      gpu->kernel = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-cpuThresh")){ // choose when to start execute on cpu
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -cpuThresh.");
        exit(1);        
      }else if(atoi(av[ai+1]) < 0){
        fprintf(stderr, "Please enter a valid value.");
        exit(1);   
      }
      gpu->cpuThresh = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-maxBlocks")){ // choose max number of blocks to open
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -maxBlocks.");
        exit(1);        
      }else if(atoi(av[ai+1]) < 0){
        fprintf(stderr, "Please enter a valid value.");
        exit(1);   
      }
      gpu->maxBlocks = atoi(av[ai+1]);
      ai += 2;
    }
  }
}

void write_bandwidth_test_out(out_str res)
{
  fprintf(stdout, "Cublas Kernel:     cuTime = %f ms, cuBandwidth = %f GB/s\n", res.cuTime, res.cuBandwidth);
  fprintf(stdout, "Reduction Kernel:  kernelTime = %f ms, kernelBandwidth= %f GB/s\n", res.kernelTime, res.kernelBandwidth);
  fprintf(stdout, "Both Kernels:      LikelihoodTime = %f ms, Likelihood_Bandwidth = %f GB/s\n", res.gpuTime, res.gpuBandwidth);
  fprintf(stdout, "Tuning Time: %f ms\n", res.tuneTime);
  fprintf(stdout, "Burn In Time: %f ms\n", res.burnTime);
  fprintf(stdout, "Metropolis Time: %f ms\n", res.mcmcTime);
  fprintf(stdout, "Sampler Time: %f ms\n", res.samplerTime);
  fprintf(stdout, "Acceptance Ratio: %f\n", res.acceptance);
  fprintf(stdout, "Ess: %f\n", res.ess);
  fprintf(stdout, "Normalised Ess: %f samples/sec\n", res.ess * 1000 / res.mcmcTime);
}

void print_normalised_sample_means(mcmc_v_str mcdata, mcmc_str mcin)
{
  //get and print mean for each dimension
  int current_idx;
  fprintf(stdout, "\nDimension: ");
  for(current_idx = 0; current_idx < mcin.ddata; current_idx++){
    fprintf(stdout, "%7d ", current_idx);
  }
  fprintf(stdout, "\nTheta:     ");
  for(current_idx = 0; current_idx < mcin.ddata; current_idx++){
    fprintf(stdout, "%+7.3f ", mcdata.sample_means[current_idx]);
  }
  fprintf(stdout, "\n\n");
}

void write_perf_out_csv(char *rootdir, out_str res, bool first)
{
  char filename[50];

  strcat(strcpy(filename,rootdir), "performance");
  write_performance_data(filename, res, first);
}


// Output csvs: Burn-In samples, Post Burn-in Samples (+normalised versions)
//              Autocorrealtion for up to lag_idx
//              Mean values for Post Burn-in Samples
void write_csv_outputs(char *rootdir, mcmc_v_str mcdata, mcmc_str mcin, 
                    sec_str sec, sec_v_str secv)
{
  fprintf(stdout, "*********************** File Output **********************\n");
  // output normal files
  output_files(rootdir, mcdata, mcin);
  // output normalised files
  output_norm_files(rootdir, mcdata, mcin);
  output_means(rootdir, mcdata, mcin);
  fprintf(stdout, "**********************************************************\n"); 

  // output autocorrelation
  fprintf(stdout, "***************** Autocorrelation Output *****************\n");
  output_autocorrelation_files(rootdir, secv, sec);
  fprintf(stdout, "**********************************************************\n"); 
}

#endif // __IO_C__