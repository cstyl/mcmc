#ifndef __MULTIPLE_RUNS_GPU_C__
#define __MULTIPLE_RUNS_GPU_C__

#include "resources.h"
#include "io.h"

#include "mcmc_gpu.h"


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

  gsl_rng *r = NULL;

  FILE *fp;
  char indir[50], outdir[50];

  read_inputs_gpu(argc, argv, &mcin, &sec, &gpu);
  mcin.impl = GPU;
  
  if(sec.fdata == 1){
    strcpy(indir, "data/runs/synthetic3d.csv");
    strcpy(outdir, "out/runs/gpu.csv");
  }else if(sec.fdata == 2){
    strcpy(indir, "data/runs/mnist.csv");
    strcpy(outdir, "out/runs_mnist/gpu.csv");
  }

  malloc_data_vectors(&data, mcin);
  malloc_sample_vectors(&mcdata, mcin);
  malloc_autocorrelation_vectors(&secv, sec);
  init_rng(&r);
  read_data(indir, ColMajor, data, mcin);

  fp=fopen(outdir,"a");
  int i;
  
  mct.rwsd = 2.38 / sqrt(mcin.ddata);

  if(mcin.tune == 1)
    tune_target_a_gpu_v2(data, r, mcin, &mct, gpu, mcdata.burn, 0.25, 100);
  else if(mcin.tune == 2)  
    tune_ess_gpu(data, r, mcin, &mct, gpu, mcdata.burn, 5000);    

  for(i=0; i<mcin.ddata; i++) 
    mcdata.burn[i] = 0;

	gpu_sampler(data, r, mcin, mct, mcdata, gpu, &res);

  printf("Acceptance = %f\n", res.acceptance);
	calculate_normalised_sample_means(mcdata, mcin);
	for(i=0; i<mcin.ddata; i++)
		if(i==mcin.ddata-1){
			fprintf(fp, "%.64f\n", mcdata.sample_means[i]);
      fprintf(stdout, "%f\n", mcdata.sample_means[i]);
		}else{
			fprintf(fp, "%.64f, ", mcdata.sample_means[i]);
      fprintf(stdout, "%f, ", mcdata.sample_means[i]);
		}

  fclose(fp);
  res.ess = get_ess(mcdata.samples, mcin.Ns, mcin.ddata, sec.lagidx, secv.circ);

  free_autocorrelation_vectors(secv);
  free_data_vectors(data, mcin);
  free_sample_vectors(mcdata);

  free_rng(r);
  return 0;
}

#endif // __MULTIPLE_RUNS_CPU_C__