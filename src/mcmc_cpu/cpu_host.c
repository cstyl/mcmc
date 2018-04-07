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
  double *x = NULL;    // data points
  double *y = NULL;    // labels
  double *sample_m = NULL;  // contains the samples in linearised space
  double *norm_sample_m = NULL;  // contains the normalised samples in linearised space
  double *burn_m = NULL; // contains the burn samples in linearised space
  double *norm_burn_m = NULL; // contains the normalised burn samples in linearised space     
  double *shift_auto_v = NULL;
  double *circ_auto_v = NULL;
  
  gsl_rng *r = NULL;
  
  data_case_t d_case = NoData;
  autocorr_case_t a_case = NoAut;
  out_case_t o_case = NoOut;
  autoc_out_case_t a_o_case = NoAutOut;

  in_struct parameters;
  out_struct results;
  double rw_sdev = 0;      // step size of random walk

  int lag_idx = 0;
  int norm = 0;
  char rootdir[50];

  double circ_autocorr_sum, shift_autocorr_sum;

  int ai = 1;
  while(ai<argc)
  {
    if(!strcmp(argv[ai],"-d")){ // choose dataset. can be from 1 to 2
      if(ai+1 >= argc){
        fprintf(stderr, "Missing argument to -d.");
        exit(1);        
      }else if((atoi(argv[ai+1]) > 2) || (atoi(argv[ai+1]) < 1)){
        fprintf(stderr, "Please enter a valid dataset value.");
        exit(1);   
      }
      d_case = atoi(argv[ai+1]);
      ai += 2;
    }else if(!strcmp(argv[ai],"-sz")){ // choose dataset size (including bias).
      if(ai+1 >= argc){
        fprintf(stderr, "Missing argument to -sz.");
        exit(1);        
      }
      parameters.Nd = atoi(argv[ai+1]);
      ai += 2;
    }else if(!strcmp(argv[ai],"-dim")){ // choose dataset dimensionality.
      if(ai+1 >= argc){
        fprintf(stderr, "Missing argument to -dim.");
        exit(1);        
      }
      parameters.d_data = atoi(argv[ai+1]);
      ai += 2;
    }else if(!strcmp(argv[ai],"-samp")){ // choose number of produced samples.
      if(ai+1 >= argc){
        fprintf(stderr, "Missing argument to -samp.");
        exit(1);        
      }
      parameters.Ns = atoi(argv[ai+1]);
      ai += 2;
    }else if(!strcmp(argv[ai],"-burn")){ // choose number of burn samples.
      if(ai+1 >= argc){
        fprintf(stderr, "Missing argument to -burn.");
        exit(1);        
      }
      parameters.burn_in = atoi(argv[ai+1]);
      ai += 2;
    }else if(!strcmp(argv[ai],"-sd")){ // choose sd for random walk.
      if(ai+1 >= argc){
        fprintf(stderr, "Missing argument to -sd.");
        exit(1);        
      }else if((atoi(argv[ai+1]) < 0)){
        fprintf(stderr, "Please enter a valid value.");
        exit(1);   
      }
      rw_sdev = atof(argv[ai+1]);
      ai += 2;
    }else if(!strcmp(argv[ai],"-lag")){ // choose lag idx
      if(ai+1 >= argc){
        fprintf(stderr, "Missing argument to -lag.");
        exit(1);        
      }else if((atoi(argv[ai+1]) < 0)){
        fprintf(stderr, "Please enter a valid value.");
        exit(1);   
      }
      lag_idx = atoi(argv[ai+1]);
      ai += 2;
    }else if(!strcmp(argv[ai],"-autoc")){ // choose lag type. can be from 0 to 3
      if(ai+1 >= argc){
        fprintf(stderr, "Missing argument to -autoc.");
        exit(1);        
      }else if((atoi(argv[ai+1]) < 0) || (atoi(argv[ai+1]) > 3)){
        fprintf(stderr, "Please enter a valid autocorrelation type value.");
        exit(1);   
      }
      a_case = atoi(argv[ai+1]);
      ai += 2;
    }else if(!strcmp(argv[ai],"-norm")){ // choose to normalise samples to make bias mean 1
      if(ai+1 >= argc){
        fprintf(stderr, "Missing argument to -norm.");
        exit(1);        
      }
      norm = atoi(argv[ai+1]);
      ai += 2;
    }else if(!strcmp(argv[ai],"-out")){
      if(ai+1 >= argc){
        fprintf(stderr, "Missing argument to -out.");
        exit(1);        
      }else if((atoi(argv[ai+1]) < 0) || (atoi(argv[ai+1]) > 3)){
        fprintf(stderr, "Please enter a valid output files case value.");
        exit(1);   
      }
      o_case = atoi(argv[ai+1]);
      ai += 2;
    }else if(!strcmp(argv[ai],"-aout")){
      if(ai+1 >= argc){
        fprintf(stderr, "Missing argument to -aout.");
        exit(1);        
      }else if((atoi(argv[ai+1]) < 0) || (atoi(argv[ai+1]) > 3)){
        fprintf(stderr, "Please enter a valid autocorrelation output files case value.");
        exit(1);   
      }
      a_o_case = atoi(argv[ai+1]);
      ai += 2;
    }
  }

  malloc_data_vectors_cpu(&x, &y, parameters.d_data, parameters.Nd);
  read_data(x, y, parameters.d_data, parameters.Nd, d_case);

  init_rng(&r);

  malloc_sample_vectors(&sample_m, &burn_m, parameters.d_data, parameters.Ns, parameters.burn_in);
  malloc_autocorrelation_vectors(&shift_auto_v, &circ_auto_v, a_case, lag_idx);

  memset(burn_m, 0, parameters.d_data*sizeof(double));

  Metropolis_Hastings_cpu(x, y, r, parameters, &results, rw_sdev, sample_m, burn_m);

  fprintf(stdout, "acceptance_ratio = %f\n", results.acc_ratio * 100);
  fprintf(stdout, "time = %dm:%ds:%dms\n", results.time_m, results.time_s, results.time_ms);
  if(d_case == 1){
    strcpy(rootdir, "out/cpu/synthetic/");
  }else if(d_case == 2){
    strcpy(rootdir, "out/cpu/mnist/");
  }

  // normalise samples
  if((norm == 1) || (o_case == Normalized) || (o_case == AllOut)){
    malloc_normalised_sample_vectors(&norm_sample_m, &norm_burn_m, 
                                      parameters.d_data, parameters.Ns, 
                                      parameters.burn_in);
    normalise_samples(burn_m, norm_burn_m, parameters.d_data, parameters.burn_in); 
    normalise_samples(sample_m, norm_sample_m, parameters.d_data, parameters.Ns);
  }
  // calculate autocorrelation
  if((a_case == Shift) || (a_case == Both) || (a_o_case == AllAut)){
    shift_autocorr_sum = shift_autocorrelation(shift_auto_v, sample_m, parameters.d_data, 
                                              parameters.Ns, lag_idx);
    results.ess_shift = parameters.Ns / (1 + 2*shift_autocorr_sum);
    fprintf(stdout, "ess_shift = %f\n", results.ess_shift);
  }

  if((a_case == Circ) || (a_case == Both) || (a_o_case == AllAut)){
    circ_autocorr_sum = circular_autocorrelation(circ_auto_v ,sample_m, parameters.d_data, 
                                                  parameters.Ns, lag_idx);
    results.ess_circular = parameters.Ns / (1 + 2*circ_autocorr_sum);
    fprintf(stdout, "ess_circular = %f\n", results.ess_circular);
  }

  // output normal files
  if((o_case == Raw) || (o_case == AllOut)){
    output_files(rootdir, sample_m, burn_m, parameters.d_data, 
                  parameters.Ns, parameters.burn_in);
  }
  // output normalised files
  if((o_case == Normalized) || (o_case == AllOut)){
    output_norm_files(rootdir, norm_sample_m, norm_burn_m, 
                      parameters.d_data, parameters.Ns, parameters.burn_in);
  }
  // output autocorrelation
  if(a_o_case != NoAutOut){
    output_autocorrelation_files(rootdir, shift_auto_v, circ_auto_v, a_o_case, lag_idx);
  }

  free_data_vectors_cpu(x, y);
  free_sample_vectors(sample_m, burn_m);
  free_norm_sample_vectors(norm_sample_m, norm_burn_m);
  free_autocorrelation_vectors(shift_auto_v, circ_auto_v, a_case);
  free_rng(r);

  return 0;
}