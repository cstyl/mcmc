#include "cpu_host.h"
#include "processing_util.h"
#include "csvparser.h"
#include "data_util.h"

// import data from csv
void read_data(int data_dim, int data, int test)
{
  CsvParser *csvparser = NULL;
  if(test == 1)
    csvparser = CsvParser_new("data/synthetic.csv", ",", 0);
  else if(test == 2)
    // csvparser = CsvParser_new("data/mnist_pca_7_9.csv", ",", 0);
    csvparser = CsvParser_new("data/norm_mnist_pca_7_9.csv", ",", 0);

  CsvRow *row;

  int datapoint=0;
  int i;

  while ((row = CsvParser_getRow(csvparser)) && (datapoint<data)) {
    char **rowFields = CsvParser_getFields(row);
    for (i = 0 ; i < CsvParser_getNumFields(row) ; i++) {
      if(i == 0){
        y[datapoint] = atof(rowFields[i]);
      }else if(i<=data_dim){
        x[datapoint*data_dim+(i-1)] = atof(rowFields[i]);
      }
    }

    CsvParser_destroy_row(row);
    datapoint++;
  }
  CsvParser_destroy(csvparser);
}

void output_norm_files(char* dir, int data_dim, 
                        int samples, int burn_samples)
{
  char filename[50];
  normalise_samples(data_dim, samples, burn_samples, get_bias_mean(data_dim, samples));
    
  strcat(strcpy(filename,dir), "norm_burn_out");
  write_data(filename, norm_burn_matrix, data_dim, burn_samples);

  strcat(strcpy(filename,dir), "norm_raw_out");
  write_data(filename, norm_sample_matrix, data_dim, samples);
}

void output_files(char* dir, int data_dim,
                  int samples, int burn_samples)
{
  char filename[50];

  strcat(strcpy(filename,dir), "burn_out");
  write_data(filename, burn_matrix, data_dim, burn_samples);

  strcat(strcpy(filename,dir), "raw_out");
  write_data(filename, sample_matrix, data_dim, samples);
}

void output_autocorrelation_files(char* dir, int auto_case, int lag)
{
  char filename[50];
  if((auto_case == 2) || (auto_case == 3)){
      strcat(strcpy(filename,dir), "circular_autocorrelation");
      write_autocorr(filename, autocorrelation_circ, lag);
  }

  if((auto_case == 1) || (auto_case == 3)){
    strcat(strcpy(filename,dir), "shift_autocorrelation");
    write_autocorr(filename, autocorrelation_shift, lag);
  }
}
/* Write the output parameters in a csv file */
void write_data(char *filename, double *data, 
                int data_dim, int data_size)
{
  fprintf(stdout, "*********************** File Output **********************\n");
  fprintf(stdout, "Creating %s.csv file\n",filename);
   
  FILE *fp;
   
  filename=strcat(filename,".csv");
   
  fp=fopen(filename,"w+");

  int i,j;
  for(i=0;i<data_size;i++){
    for(j=0;j<data_dim;j++)
    {
      if(j<data_dim-1)
        fprintf(fp, "%f, ", data[i*data_dim+j]);
      else
        fprintf(fp, "%f\n", data[i*data_dim+j]);
    }
  }

  fclose(fp);
  fprintf(stdout, "%s file created\n",filename);
  fprintf(stdout, "**********************************************************\n"); 
}

/* Write the output parameters in a csv file */
void write_autocorr(char *filename, double *autocorrelation, int lag_idx)
{
  fprintf(stdout, "***************** Autocorrelation Output *****************\n");
  fprintf(stdout, "Creating %s.csv file\n",filename);
   
  FILE *fp;
   
  filename=strcat(filename,".csv");
   
  fp=fopen(filename,"w+");

  fprintf(fp,"lag_k, autocorrelation\n");

  int i;
  for(i=0;i<lag_idx;i++){
    fprintf(fp, "%d, %f\n", i, autocorrelation[i]);
  }

  fclose(fp);
  fprintf(stdout, "%s file created\n",filename); 
  fprintf(stdout, "**********************************************************\n");
}