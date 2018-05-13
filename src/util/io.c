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
    }else if(!strcmp(av[ai],"-sz")){ // choose dataset size (including bias).
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -sz.");
        exit(1);        
      }
      mcin->Nd = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-dim")){ // choose dataset dimensionality.
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
    }else if(!strcmp(av[ai],"-autoc")){ // choose lag type. can be from 0 to 3
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -autoc.");
        exit(1);        
      }else if((atoi(av[ai+1]) < 0) || (atoi(av[ai+1]) > 3)){
        fprintf(stderr, "Please enter a valid autocorrelation type value.");
        exit(1);   
      }
      sec->fauto = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-norm")){ // choose to normalise samples to make bias mean 1
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -norm.");
        exit(1);        
      }
      sec->fnorm = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-out")){
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -out.");
        exit(1);        
      }else if((atoi(av[ai+1]) < 0) || (atoi(av[ai+1]) > 3)){
        fprintf(stderr, "Please enter a valid output files case value.");
        exit(1);   
      }
      sec->fout = atoi(av[ai+1]);
      ai += 2;
    }else if(!strcmp(av[ai],"-aout")){
      if(ai+1 >= an){
        fprintf(stderr, "Missing argument to -aout.");
        exit(1);        
      }else if((atoi(av[ai+1]) < 0) || (atoi(av[ai+1]) > 3)){
        fprintf(stderr, "Please enter a valid autocorrelation output files case value.");
        exit(1);   
      }
      sec->fautoout = atoi(av[ai+1]);
      ai += 2;
    }
  }
}

void write_outputs(char *rootdir, mcmc_v_str mcdata, 
                    mcmc_str mcin, sec_str sec, 
                    out_str results)
{
  sec_v_str secv;
  double circ_sum, shift_sum;
  double bias_mu;
  malloc_autocorrelation_vectors(&secv, sec);

  fprintf(stdout, "acceptance_ratio = %f\n", results.acc_ratio * 100);
  fprintf(stdout, "time = %dm:%ds:%dms\n", results.time_m, results.time_s, results.time_ms);
  
  // normalise samples
  if((sec.fnorm == 1) || (sec.fout == 2) || (sec.fout == 3)){
    malloc_normalised_sample_vectors(&mcdata, mcin);
    bias_mu = get_bias_mean(mcdata.samples, mcin.ddata, mcin.Ns);
    normalise_samples(mcdata.burn, mcdata.nburn, mcin.ddata, mcin.burnin, bias_mu); 
    normalise_samples(mcdata.samples, mcdata.nsamples, mcin.ddata, mcin.Ns, bias_mu);
    //get and print mean for each dimension
    int current_idx;
    fprintf(stdout, "\nDimension: ");
    for(current_idx = 0; current_idx < mcin.ddata; current_idx++){
      fprintf(stdout, "%7d ", current_idx);
    }
    fprintf(stdout, "\nTheta:     ");
    for(current_idx = 0; current_idx < mcin.ddata; current_idx++){
      mcdata.sample_means[current_idx] = get_dim_mean(mcdata.nsamples, mcin.ddata, current_idx, mcin.Ns);
      fprintf(stdout, "%+7.3f ", mcdata.sample_means[current_idx]);
    }
    fprintf(stdout, "\n\n");
  }
  // calculate autocorrelation
  if((sec.fauto == 1) || (sec.fauto == 3) || (sec.fautoout == 3)){
    shift_sum = shift_autocorrelation(secv.shift, mcdata.samples, 
                                      mcin.ddata, mcin.Ns, sec.lagidx);
    results.ess_shift = mcin.Ns / (1 + 2*shift_sum);
    fprintf(stdout, "ess_shift = %f\n", results.ess_shift);
  }

  if((sec.fauto == 2) || (sec.fauto == 3) || (sec.fautoout == 3)){
    circ_sum = circular_autocorrelation(secv.circ ,mcdata.samples, 
                                        mcin.ddata, mcin.Ns, sec.lagidx);
    results.ess_circular = mcin.Ns / (1 + 2*circ_sum);
    fprintf(stdout, "ess_circular = %f\n", results.ess_circular);
  }

  fprintf(stdout, "*********************** File Output **********************\n");
  // output normal files
  if((sec.fout == 1) || (sec.fout == 3)){
    output_files(rootdir, mcdata, mcin);
  }
  // output normalised files
  if((sec.fout == 2) || (sec.fout == 3)){
    output_norm_files(rootdir, mcdata, mcin);
    output_means(rootdir, mcdata, mcin);
  }
  fprintf(stdout, "**********************************************************\n"); 

  // output autocorrelation
  if(sec.fautoout != 0){
    fprintf(stdout, "***************** Autocorrelation Output *****************\n");
    output_autocorrelation_files(rootdir, secv, sec);
    fprintf(stdout, "**********************************************************\n"); 
  }
  
  free_norm_sample_vectors(mcdata);
  free_autocorrelation_vectors(secv, sec);
}