#ifndef __CPU__TUNNING_C__
#define __CPU__TUNNING_C__

#include "cpu_host.h"

void export_tunning_test(char *dir, double *autocorrelation, double *samples, double *burn_samples, int Nd, int dim, int burn, int lagidx);

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

  char *dir = (char*) malloc(70*sizeof(char));

  clock_t start, stop;

  init_rng(&r);

  read_inputs(argc, argv, &mcin, &sec);
  mcin.impl = CPU;
  
  if(sec.fdata == 1){
    strcpy(indir, "data/tunning_cpu/synthetic3d.csv");
    strcpy(outdir, "out/tunning_cpu/synthetic_");
  }else if(sec.fdata == 2){
    strcpy(indir, "data/tunning_cpu/mnist.csv");
    strcpy(outdir, "out/tunning_cpu/mnist_");
  }

  malloc_data_vectors(&data, mcin);
  malloc_sample_vectors(&mcdata, mcin);
  malloc_autocorrelation_vectors(&secv, sec);
  
  read_data(indir, ColMajor, data, mcin);

  mct.rwsd = 2.38 / sqrt(mcin.ddata);
  
  fp=fopen(outdir,"w+");
  fprintf(fp, "version, ess, t(ms), ess/t(s), acceptance\n");  
  int i;
  for(i=0; i<mcin.ddata; i++) mcdata.burn[i] = 0;
  
  // tune for target ratio
  tune_target_a_cpu_v2(data, r, mcin, &mct, mcdata.burn, 0.25, 100);   
  
  start  = clock();
  cpu_sampler(data, r, mcin, mct, mcdata, &res);
  stop = clock() - start;

  res.samplerTime = stop * 1000 / CLOCKS_PER_SEC;  // sampler time in ms
  res.ess = get_ess(mcdata.samples, mcin.Ns, mcin.ddata, sec.lagidx, secv.circ);

  // export autocorrelation, samples & burn samples in out/tunning_cpu/synthetic_tunned_target
  strcpy(dir, outdir);
  strcat(dir,"tunned_target_");

  export_tunning_test(dir, secv.circ, mcdata.samples, mcdata.burn, mcin.Ns, mcin.ddata, mcin.burnin, sec.lagidx);

  fprintf(fp, "tunned_target, %f, %f, %f, %f\n", res.ess, res.samplerTime, res.ess/(res.samplerTime*1000), res.acceptance);


  mct.rwsd = 2.38 / sqrt(mcin.ddata);
  tune_target_a_cpu_v2(data, r, mcin, &mct, mcdata.burn, 0.90, 100);

  start  = clock();
  cpu_sampler(data, r, mcin, mct, mcdata, &res);
  stop = clock() - start;

  res.samplerTime = stop * 1000 / CLOCKS_PER_SEC;  // sampler time in ms
  res.ess = get_ess(mcdata.samples, mcin.Ns, mcin.ddata, sec.lagidx, secv.circ);
  printf("ess done\n");

  strcpy(dir, outdir);
  strcat(dir,"small_sd_");

  export_tunning_test(dir, secv.circ, mcdata.samples, mcdata.burn, mcin.Ns, mcin.ddata, mcin.burnin, sec.lagidx);

  fprintf(fp, "small_sd, %f, %f, %f, %f\n", res.ess, res.samplerTime, res.ess/(res.samplerTime*1000), res.acceptance);


  mct.rwsd = 2.38 / sqrt(mcin.ddata);
  tune_target_a_cpu_v2(data, r, mcin, &mct, mcdata.burn, 0.03, 100);

  start  = clock();
  cpu_sampler(data, r, mcin, mct, mcdata, &res);
  stop = clock() - start;

  res.samplerTime = stop * 1000 / CLOCKS_PER_SEC;  // sampler time in ms
  res.ess = get_ess(mcdata.samples, mcin.Ns, mcin.ddata, sec.lagidx, secv.circ);

  strcpy(dir, outdir);
  strcat(dir,"large_sd_");

  export_tunning_test(dir, secv.circ, mcdata.samples, mcdata.burn, mcin.Ns, mcin.ddata, mcin.burnin, sec.lagidx);

  fprintf(fp, "large_sd, %f, %f, %f, %f\n", res.ess, res.samplerTime, res.ess/(res.samplerTime*1000), res.acceptance);


  mct.rwsd = 2.38 / sqrt(mcin.ddata);
  // tune for ess
  tune_ess_cpu(data, r, mcin, &mct, mcdata.burn, 5000); 

  start  = clock();
  cpu_sampler(data, r, mcin, mct, mcdata, &res);
  stop = clock() - start;

  res.samplerTime = stop * 1000 / CLOCKS_PER_SEC;  // sampler time in ms
  res.ess = get_ess(mcdata.samples, mcin.Ns, mcin.ddata, sec.lagidx, secv.circ);

  strcpy(dir, outdir);
  strcat(dir,"tune_ess_");

  export_tunning_test(dir, secv.circ, mcdata.samples, mcdata.burn, mcin.Ns, mcin.ddata, mcin.burnin, sec.lagidx);

  fprintf(fp, "tune_ess, %f, %f, %f, %f\n", res.ess, res.samplerTime, res.ess/(res.samplerTime*1000), res.acceptance);

  fclose(fp);

  free(dir);
  free_autocorrelation_vectors(secv);
  free_data_vectors(data, mcin);
  free_sample_vectors(mcdata);
  free_rng(r);

  return 0;
}

#endif // __CPU__TUNNING_C__

void export_tunning_test(char *dir, double *autocorrelation, double *samples, double *burn_samples, int Ns, int dim, int burn, int lagidx)
{
  char *filename = (char*) malloc(70*sizeof(char));
  FILE *fp, *fp1, *fp2;
  
  strcpy(filename, dir);
  strcat(filename,"autocorrelation.csv");

  printf("%s\n", filename);
  fp=fopen(filename,"w+");

  int i,j;

  for(j=0;j<lagidx;j++)
  {
    fprintf(fp, "%d, %f\n", j, autocorrelation[j]);
  }

  fclose(fp);

  strcpy(filename, dir);
  strcat(filename,"samples.csv");

  fp1=fopen(filename,"w+");

  for(i=0;i<Ns;i++){
    for(j=0;j<dim;j++)
    {
      if(j<dim-1)
        fprintf(fp1, "%f, ", samples[i*dim+j]);
      else
        fprintf(fp1, "%f\n", samples[i*dim+j]);
    }
  }

  fclose(fp1);

  strcpy(filename, dir);
  strcat(filename,"burn_samples.csv");

  fp2=fopen(filename,"w+");

  for(i=0;i<burn;i++){
    for(j=0;j<dim;j++)
    {
      if(j<dim-1)
        fprintf(fp2, "%f, ", burn_samples[i*dim+j]);
      else
        fprintf(fp2, "%f\n", burn_samples[i*dim+j]);
    }
  }

  fclose(fp2);
  free(filename);
}