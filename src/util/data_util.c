#ifndef __DATA_UTIL_C__
#define __DATA_UTIL_C__

#include "data_util.h"

void read_data(char* dir, int store, data_str data, mcmc_str mcin)
{
  CsvParser *csvparser = NULL;

  csvparser = CsvParser_new(dir, ",", 0);

  CsvRow *row;

  int datapoint=0;
  int i;
  int8_t label = 1;
  fprintf(stdout, "Importing data. ");
  while ((row = CsvParser_getRow(csvparser)) && (datapoint<mcin.Nd)) {
    char **rowFields = CsvParser_getFields(row);
    if(mcin.impl == MP)
    {
      data.zlabels[datapoint] = 0;      // initialize z label to 0
      data.zidx[datapoint] = datapoint; // get the index of the datapoint
    }
    for (i = 0 ; i < CsvParser_getNumFields(row) ; i++) {
      if(i == 0){
        if(atof(rowFields[i]) > 0){
          label = 1;
        }else{
          label = -1;
        }  
      }else if(i <= mcin.ddata){
        if(store==RowMajor){
          data.data[datapoint*mcin.ddata+(i-1)] = -label * atof(rowFields[i]);
        }else if(store==ColMajor){
          data.data[(i-1)*mcin.Nd+datapoint] = -label * atof(rowFields[i]);
        }
      }
    }

    CsvParser_destroy_row(row);
    datapoint++;
  }
  fprintf(stdout, "Done\n");
  CsvParser_destroy(csvparser);
}

void output_norm_files(char* dir, mcmc_v_str mcdata, mcmc_str mcin)
{
  char filename[50];
  
  strcat(strcpy(filename,dir), "norm_burn_out");
  write_data(filename, mcdata.nburn, mcin.burnin, mcin.ddata);

  strcat(strcpy(filename,dir), "norm_raw_out");
  write_data(filename, mcdata.nsamples, mcin.Ns, mcin.ddata);
}

void output_means(char* dir, mcmc_v_str mcdata, mcmc_str mcin)
{
  char filename[50];
  
  strcat(strcpy(filename,dir), "mean_out");
  write_data(filename, mcdata.sample_means, mcin.ddata, 1);
}

void output_files(char* dir, mcmc_v_str mcdata, mcmc_str mcin)
{
  char filename[50];

  strcat(strcpy(filename,dir), "burn_out");
  write_data(filename, mcdata.burn, mcin.burnin, mcin.ddata);

  strcat(strcpy(filename,dir), "raw_out");
  write_data(filename, mcdata.samples, mcin.Ns, mcin.ddata);
}

void output_autocorrelation_files(char* dir, sec_v_str secv, sec_str sec)
{
  char filename[50];

  // strcat(strcpy(filename,dir), "shift_autocorrelation");
  // write_autocorr(filename, secv.shift, sec);

  strcat(strcpy(filename,dir), "circular_autocorrelation");
  write_autocorr(filename, secv.circ, sec);
}

void write_performance_data(char *filename, out_str res, bool first)
{
  fprintf(stdout, "Opening %s.csv file. ",filename);
  FILE *fp;
   
  filename=strcat(filename,".csv");
   
  fp=fopen(filename,"a");

  if(first){  // create the header of the csv
    fprintf(fp, "Nd, dim, samples, burnSamples, sd, device, samplerTime, mcmcTime, tuneTime, burnTime, ");
    fprintf(fp, "cpuTime, kernel, blocksz, gpuTime, cuTime, kernelTime, ");
    fprintf(fp, "gpuBandwidth, cuBandwidth, kernelBandwidth, acceptance, ess\n");
  }else{
    fprintf(fp, "%d, %d, %d, %d, %f, %d, %f, %f, %f, %f, ", res.Nd, res.dim, res.samples, res.burnSamples, 
            res.sd, res.device, res.samplerTime, res.mcmcTime, res.tuneTime, res.burnTime);
    fprintf(fp, "%f, %d, %d, %f, %f, %f, ", res.cpuTime, res.kernel, res.blocksz,
            res.gpuTime, res.cuTime, res.kernelTime);
    fprintf(fp, "%f, %f, %f, %f, %f\n", res.gpuBandwidth, res.cuBandwidth, res.kernelBandwidth,
            res.acceptance, res.ess);
  }


  fclose(fp);
  fprintf(stdout, "File appended.\n");
}

/* Write the output parameters in a csv file */
void write_data(char *filename, double *data, int sz, int dim)
{
  fprintf(stdout, "Creating %s.csv file. ",filename);
   
  FILE *fp;
   
  filename=strcat(filename,".csv");
   
  fp=fopen(filename,"w+");

  int i,j;
  for(i=0;i<sz;i++){
    for(j=0;j<dim;j++)
    {
      if(j<dim-1)
        fprintf(fp, "%f, ", data[i*dim+j]);
      else
        fprintf(fp, "%f\n", data[i*dim+j]);
    }
  }

  fclose(fp);
  fprintf(stdout, "File created\n");
}

// void write_performance_gpu(char *filename, out_str res)
// {
//   fprintf(stdout, "Creating %s.csv file. ",filename);
   
//   FILE *fp;
   
//   filename=strcat(filename,".csv");
   
//   fp=fopen(filename,"w+");

//   int i,j;
//   for(i=0;i<sz;i++){
//     for(j=0;j<dim;j++)
//     {
//       if(j<dim-1)
//         fprintf(fp, "%f, ", data[i*dim+j]);
//       else
//         fprintf(fp, "%f\n", data[i*dim+j]);
//     }
//   }

//   fclose(fp);
//   fprintf(stdout, "File created\n");
// }

/* Write the output parameters in a csv file */
void write_autocorr(char *filename, double *autocorrelation, sec_str sec)
{
  fprintf(stdout, "Creating %s.csv file. ",filename);
   
  FILE *fp;
   
  filename=strcat(filename,".csv");
   
  fp=fopen(filename,"w+");

  fprintf(fp,"lag_k, autocorrelation\n");

  int i;
  for(i=0;i<sec.lagidx;i++){
    fprintf(fp, "%d, %f\n", i, autocorrelation[i]);
  }

  fclose(fp);
  fprintf(stdout, "File created\n"); 
}

#endif // __DATA_UTIL_C__