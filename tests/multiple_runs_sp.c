#ifndef __MULTIPLE_RUNS_SP_C__
#define __MULTIPLE_RUNS_SP_C__

#include "resources.h"
#include "io.h"

#include "mcmc_gpu_sp.h"
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
  mcin.impl = SP;
  
  if(sec.fdata == 1){
    strcpy(indir, "data/runs/synthetic3d.csv");
    strcpy(outdir, "out/runs/sp.csv");
  }else if(sec.fdata == 2){
    strcpy(indir, "data/runs/mnist.csv");
    strcpy(outdir, "out/runs_mnist/sp.csv");
  }

  malloc_data_vectors_sp(&data, mcin);
  malloc_sample_vectors_sp(&mcdata, mcin);
  malloc_autocorrelation_vectors_sp(&secv, sec);
  init_rng(&r);
  read_data_sp(indir, ColMajor, data, mcin);

  fp=fopen(outdir,"a");
  int i;
  
  mct.rwsdf = 2.38 / (float) (sqrt(mcin.ddata));

  if(mcin.tune == 1)
    tune_target_a_sp_v2(data, r, mcin, &mct, gpu, mcdata.burnf, 0.25, 100);
  else if(mcin.tune == 2)  
    tune_ess_sp(data, r, mcin, &mct, gpu, mcdata.burnf, 5000);    

  for(i=0; i<mcin.ddata; i++) 
    mcdata.burnf[i] = 0;

	sp_sampler(data, r, mcin, mct, mcdata, gpu, &res);

  printf("Acceptance = %f\n", res.acceptance);
	calculate_normalised_sample_means_sp(mcdata, mcin);

	for(i=0; i<mcin.ddata; i++)
		if(i==mcin.ddata-1){
			fprintf(fp, "%.32f\n", mcdata.sample_meansf[i]);
      fprintf(stdout, "%f\n", mcdata.sample_meansf[i]);
		}else{
			fprintf(fp, "%.32f, ", mcdata.sample_meansf[i]);
      fprintf(stdout, "%f, ", mcdata.sample_meansf[i]);
		}

  fclose(fp);
  res.ess = get_ess_sp(mcdata.samplesf, mcin.Ns, mcin.ddata, sec.lagidx, secv.circf);

  free_autocorrelation_vectors_sp(secv);
  free_data_vectors_sp(data, mcin);
  free_sample_vectors_sp(mcdata);

  free_rng(r);
  return 0;
}

#endif // __MULTIPLE_RUNS_CPU_C__