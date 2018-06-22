#ifndef __MULTIPLE_RUNS_CPU_C__
#define __MULTIPLE_RUNS_CPU_C__

#include "resources.h"
#include "io.h"

#include "mcmc_cpu.h"


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
  
  FILE *fp;
  char indir[50], outdir[50];

  read_inputs(argc, argv, &mcin, &sec);
  mcin.impl = CPU;
  
  if(sec.fdata == 1){
    strcpy(indir, "data/runs/synthetic3d.csv");
    strcpy(outdir, "out/runs/cpu.csv");
  }else if(sec.fdata == 2){
    strcpy(indir, "data/runs/mnist.csv");
    strcpy(outdir, "out/runs_mnist/cpu.csv");
  }

  malloc_data_vectors(&data, mcin);
  malloc_sample_vectors(&mcdata, mcin);
  malloc_autocorrelation_vectors(&secv, sec);
  init_rng(&r);
  read_data(indir, ColMajor, data, mcin);

  fp=fopen(outdir,"w+");
  int iter, i;

  for(i=0; i<mcin.ddata; i++) 
		mcdata.burn[i] = 0;
  
  mct.rwsd = 2.38 / sqrt(mcin.ddata);
  if(mcin.tune == 1)
    tune_target_a_cpu_v2(data, r, mcin, &mct, mcdata.burn, 0.25, 40);
  else if(mcin.tune == 2)  
    tune_ess_cpu(data, r, mcin, &mct, mcdata.burn, 5000);    

  for(iter=0; iter<3000; iter++)
  {
  	printf("iteration: %d\n", iter);
	  for(i=0; i<mcin.ddata; i++) 
			mcdata.burn[i] = 0;

		cpu_sampler(data, r, mcin, mct, mcdata, &res);

  	calculate_normalised_sample_means(mcdata, mcin);
  	for(i=0; i<mcin.ddata; i++)
  		if(i==mcin.ddata-1){
  			fprintf(fp, "%.64f\n", mcdata.sample_means[i]);
  		}else{
  			fprintf(fp, "%.64f, ", mcdata.sample_means[i]);
  		}
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