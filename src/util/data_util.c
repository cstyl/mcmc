// #include "cpu_host.h"
// #include "gpu_host.h"
#include "data_util.h"

// import data from csv
void read_data(data_str data, mcmc_str mcin, sec_str sec)
{
  CsvParser *csvparser = NULL;
  if(sec.fdata == 1)
    csvparser = CsvParser_new("data/synthetic.csv", ",", 0);
  else if(sec.fdata == 2)
    // csvparser = CsvParser_new("data/mnist_pca_7_9.csv", ",", 0);
    csvparser = CsvParser_new("data/norm_mnist_pca_7_9.csv", ",", 0);

  CsvRow *row;

  int datapoint=0;
  int i;
  fprintf(stdout, "Importing data. ");
  while ((row = CsvParser_getRow(csvparser)) && (datapoint<mcin.Nd)) {
    char **rowFields = CsvParser_getFields(row);
    for (i = 0 ; i < CsvParser_getNumFields(row) ; i++) {
      if(i == 0){
        if(atof(rowFields[i]) > 0){
          data.labels[datapoint] = 1;
        }else{
          data.labels[datapoint] = -1;
        }  
      }else if(i <= mcin.ddata){
        data.data[datapoint*mcin.ddata+(i-1)] = atof(rowFields[i]);
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

  if((sec.fauto == 1) || (sec.fauto == 3)){
    strcat(strcpy(filename,dir), "shift_autocorrelation");
    write_autocorr(filename, secv.shift, sec);
  }

  if((sec.fauto == 2) || (sec.fauto == 3)){
      strcat(strcpy(filename,dir), "circular_autocorrelation");
      write_autocorr(filename, secv.circ, sec);
  }
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