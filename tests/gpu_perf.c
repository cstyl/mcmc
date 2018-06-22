/*
 * Implementation of MCMC Metropolis-Hastings Algorithm on CPU
 * Aim: Obtain a sequence of RANDOM samples from a probability distribution
 * from which direct sampling is difficult
 * The sequence can be used to approximate the distribution (eg histogram) 
 * or compute an integral (eg expected value)
 */
#ifndef __GPU_PERF_C__
#define __GPU_PERF_C__

#include "gpu_host.h"

void fill_array(int *array_in, sec_str sec);

int main(int argc, char * argv[])
{
  data_str data;

  mcmc_str mcin;
  mcmc_tune_str mct;
  mcmc_v_str mcdata;

  gpu_v_str gpu;

  sec_str sec;
  sec_v_str secv;
  out_str res;
  FILE *fp;
  gsl_rng *r = NULL;
  
  char indir[50], outdir[50];
  clock_t start, stop;

  init_rng(&r);

  read_inputs_gpu(argc, argv, &mcin, &sec, &gpu);
  mcin.impl = GPU;

  int Nd[7];
	int blockArr[] = {64, 128, 256, 512, 1024};

  strcpy(indir, "data/gpu_performance/synthetic.csv");  
  if(sec.fdata == 3){
    strcpy(outdir, "out/gpu_performance/gpu_10.csv");
  }else if(sec.fdata==4){
    strcpy(outdir, "out/gpu_performance/gpu_20.csv");
  }else if(sec.fdata == 5){
    strcpy(outdir, "out/gpu_performance/gpu_50.csv");
  }else if(sec.fdata == 6){
    strcpy(outdir, "out/gpu_performance/gpu_100.csv");
  }else if(sec.fdata == 7){
    strcpy(outdir, "out/gpu_performance/gpu_200.csv");
  }

  malloc_data_vectors(&data, mcin);
  malloc_sample_vectors(&mcdata, mcin);
  malloc_autocorrelation_vectors(&secv, sec);
  
  read_data(indir, ColMajor, data, mcin);
  fill_array(Nd, sec);

  int i, ndIdx, blockIdx;
  fp=fopen(outdir,"w+");
  fprintf(fp, "Nd, Dim, samplerTime(ms), burnTime(ms), mcmcTime(ms), Bandwidth(GBs/s), Block Size\n"); 

  for(ndIdx=0; ndIdx<7; ndIdx++)
  {
  	for(blockIdx=0; blockIdx<5; blockIdx++)
  	{
	    mcin.Nd = Nd[ndIdx];
	  	gpu.maxThreads = blockArr[blockIdx];
	    mct.rwsd = 2.38 / sqrt(mcin.ddata);
	    for(i=0; i<mcin.ddata; i++) mcdata.burn[i] = 0;

	    printf("Idx: %d...Starting run for Nd:%d, dim:%d, block:%d\n", ndIdx, mcin.Nd, mcin.ddata, gpu.maxThreads);
	    start  = clock();
			gpu_sampler(data, r, mcin, mct, mcdata, gpu, &res);
	    stop = clock() - start;
	    printf("Idx: %d... Run for Nd:%d, dim:%d, block:%d Completed..\n", ndIdx, mcin.Nd, mcin.ddata, gpu.maxThreads);
	    res.samplerTime = stop * 1000 / CLOCKS_PER_SEC;  // sampler time in ms

	    fprintf(fp, "%d, %d, %.32f, %.32f, %.32f, %.32f, %d\n", mcin.Nd, mcin.ddata, res.samplerTime, res.burnTime, res.mcmcTime, res.gpuBandwidth, gpu.maxThreads);
	  }
  }

  fclose(fp);

  free_autocorrelation_vectors(secv);
  free_data_vectors(data, mcin);
  free_sample_vectors(mcdata);
  free_rng(r);

  return 0;
}


void fill_array(int *array_in, sec_str sec)
{
	if(sec.fdata == 3){
		array_in[0] = 1280;
		array_in[1] = 6400;
		array_in[2] = 13100;
		array_in[3] = 65536;
		array_in[4] = 262144;
		array_in[5] = 655360;
		array_in[6] =	1310720;
  }else if(sec.fdata==4){
		array_in[0] = 640;
		array_in[1] = 3200;
		array_in[2] = 6553;
		array_in[3] = 32768;
		array_in[4] = 131072;
		array_in[5] = 327680;
		array_in[6] =	655360;
  }else if(sec.fdata == 5){
		array_in[0] = 256; 		
		array_in[1] = 1280; 		
		array_in[2] = 2621; 		
		array_in[3] = 13107; 		
		array_in[4] = 52428; 		
		array_in[5] = 131072; 		
		array_in[6] = 262144;
  }else if(sec.fdata == 6){
		array_in[0] = 128;
		array_in[1] = 640;
		array_in[2] = 1310;
		array_in[3] = 6553;
		array_in[4] = 26214;
		array_in[5] = 65536;
		array_in[6] = 131072;
  }else{
		array_in[0] = 64;
		array_in[1] = 320;
		array_in[2] = 655;
		array_in[3] = 3276;
		array_in[4] = 13107;
		array_in[5] = 32768;
		array_in[6] = 65536;
  }
}
#endif // __LARGE_DATA_CPU_C__