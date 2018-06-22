#ifndef __COMPARE_TUNERS_C__
#define __COMPARE_TUNERS_C__

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
  FILE *fp;
  gsl_rng *r = NULL;
  gpu_v_str gpu;  
  char indir[50], outdir[50];

  init_rng(&r);

  read_inputs_gpu(argc, argv, &mcin, &sec, &gpu);
  mcin.impl = CPU;

  if(sec.fdata==1){
    strcpy(indir, "data/compare_tuners/synthetic.csv");  
    strcpy(outdir, "out/compare_tuners/synthetic.csv");
  }else{
    strcpy(indir, "data/compare_tuners/mnist.csv");  
    strcpy(outdir, "out/compare_tuners/mnist.csv");
  }

  malloc_data_vectors(&data, mcin);
  malloc_sample_vectors(&mcdata, mcin);
  malloc_autocorrelation_vectors(&secv, sec);
  
  read_data(indir, ColMajor, data, mcin);

  int i;
  fp=fopen(outdir,"a");

  mct.rwsd = 2.38 / sqrt(mcin.ddata);
  for(i=0; i<mcin.ddata; i++) mcdata.burn[i] = 0;

  printf("Tuning For Target: Starting run for Nd:%d, dim:%d\n",mcin.Nd, mcin.ddata);
    tune_target_a_gpu_v2(data, r, mcin, &mct, gpu, mcdata.burn, 0.25, 100);
  gpu_sampler(data, r, mcin, mct, mcdata, gpu, &res);
  printf("Run for Nd:%d, dim:%d Completed..\n", mcin.Nd, mcin.ddata);
  res.ess = get_ess(mcdata.samples, mcin.Ns, mcin.ddata, sec.lagidx, secv.circ);
  fprintf(fp, "%d, %d, %.32f, ", mcin.ddata, mcin.Nd, res.ess);

  mct.rwsd = 2.38 / sqrt(mcin.ddata);
  printf("Tuning For ESS: Starting run for Nd:%d, dim:%d\n",mcin.Nd, mcin.ddata);
    tune_ess_gpu(data, r, mcin, &mct, gpu, mcdata.burn, 5000); 
  gpu_sampler(data, r, mcin, mct, mcdata, gpu, &res);
  printf("Run for Nd:%d, dim:%d Completed..\n", mcin.Nd, mcin.ddata);
  res.ess = get_ess(mcdata.samples, mcin.Ns, mcin.ddata, sec.lagidx, secv.circ);
  fprintf(fp, "%.32f\n", res.ess);

  fclose(fp);

  free_autocorrelation_vectors(secv);
  free_data_vectors(data, mcin);
  free_sample_vectors(mcdata);
  free_rng(r);

  return 0;
}

#endif // __COMPARE_TUNERS_C__
// #ifndef __COMPARE_TUNERS_C__
// #define __COMPARE_TUNERS_C__

// #include "cpu_host.h"

// int main(int argc, char * argv[])
// {
//   data_str data;

//   mcmc_str mcin;
//   mcmc_tune_str mct;
//   mcmc_v_str mcdata;

//   sec_str sec;
//   sec_v_str secv;
//   out_str res;
//   FILE *fp;
//   gsl_rng *r = NULL;

//   char indir[50], outdir[50];

//   init_rng(&r);

//   read_inputs(argc, argv, &mcin, &sec);
//   mcin.impl = CPU;

//   if(sec.fdata==1){
//     strcpy(indir, "data/compare_tuners/synthetic.csv");  
//     strcpy(outdir, "out/compare_tuners/synthetic.csv");
//   }else{
//     strcpy(indir, "data/compare_tuners/mnist.csv");  
//     strcpy(outdir, "out/compare_tuners/mnist.csv");
//   }

//   malloc_data_vectors(&data, mcin);
//   malloc_sample_vectors(&mcdata, mcin);
//   malloc_autocorrelation_vectors(&secv, sec);
  
//   read_data(indir, ColMajor, data, mcin);

//   int i;
//   fp=fopen(outdir,"a");

//   mct.rwsd = 2.38 / sqrt(mcin.ddata);
//   for(i=0; i<mcin.ddata; i++) mcdata.burn[i] = 0;

//   printf("Tuning For Target: Starting run for Nd:%d, dim:%d\n",mcin.Nd, mcin.ddata);
//   tune_target_a_cpu_v2(data, r, mcin, &mct, mcdata.burn, 0.25, 100);
//   cpu_sampler(data, r, mcin, mct, mcdata, &res);
//   printf("Run for Nd:%d, dim:%d Completed..\n", mcin.Nd, mcin.ddata);
//   res.ess = get_ess(mcdata.samples, mcin.Ns, mcin.ddata, sec.lagidx, secv.circ);
//   fprintf(fp, "%d, %d, %.32f, ", mcin.ddata, mcin.Nd, res.ess);

//   mct.rwsd = 2.38 / sqrt(mcin.ddata);
//   printf("Tuning For ESS: Starting run for Nd:%d, dim:%d\n",mcin.Nd, mcin.ddata);
//   tune_ess_cpu(data, r, mcin, &mct, mcdata.burn, 5000); 
//   cpu_sampler(data, r, mcin, mct, mcdata, &res);
//   printf("Run for Nd:%d, dim:%d Completed..\n", mcin.Nd, mcin.ddata);
//   res.ess = get_ess(mcdata.samples, mcin.Ns, mcin.ddata, sec.lagidx, secv.circ);
//   fprintf(fp, "%.32f\n", res.ess);

//   fclose(fp);

//   free_autocorrelation_vectors(secv);
//   free_data_vectors(data, mcin);
//   free_sample_vectors(mcdata);
//   free_rng(r);

//   return 0;
// }

// #endif // __COMPARE_TUNERS_C__