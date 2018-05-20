#include "performanceTest.h"

const int KERNELS = 4;


int main(int argc, char *argv[])
{
    data_str data;

    mcmc_str mcin;
    mcmc_tune_str mct;
    mcmc_v_str mcdata;

    sec_str sec;
    sec_v_str secv;

    gpu_v_str gpu;
    out_str res;

    gsl_rng *r = NULL;
    char indir[50], outdir[50];
    clock_t start, stop;

    init_rng(&r);

    int dim_idx, kernel_idx, block_idx;

    read_inputs(argc, argv, &mcin, &sec);

    if(sec.fdata == 1){
        strcpy(indir, "data/performance/synthetic.csv");
        strcpy(outdir, "out/performance/synthetic/");
    }else if(sec.fdata == 2){
        strcpy(indir, "data/performance/mnist.csv");
        strcpy(outdir, "out/performance/mnist/");
    }
    
    if(sec.first == 1){ write_perf_out_csv(outdir, res, true); }  //create the header of the csv
    
    malloc_data_vectors(&data, mcin);

    read_data(indir, ColMajor, data, mcin);

    mct.rwsd = 2.38 / sqrt(mcin.ddata);

    for(kernel_idx=0; kernel_idx<KERNELS; kernel_idx++)
    {
        gpu.size = mcin.Nd;
        gpu.kernel = kernel_idx;
        gpu.maxBlocks = 64;
        gpu.cpuThresh = 32;

        for(block_idx=1024; block_idx>16; block_idx/=2)
        {
            gpu.maxThreads = block_idx;
            malloc_sample_vectors(&mcdata, mcin);
            malloc_autocorrelation_vectors(&secv, sec);

            for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++){ mcdata.burn[dim_idx] = 0; } 

            start  = clock();
            gpu_sampler(data, r, mcin, &mct, mcdata, gpu, &res);
            stop = clock() - start;

            res.samplerTime = stop * 1000 / CLOCKS_PER_SEC;  // sampler time in ms
            res.ess = get_ess(mcdata.samples, mcin.Ns, mcin.ddata, sec.lagidx, secv.circ);

            res.device = 2;     //  gpu
            res.Nd = mcin.Nd;
            res.dim = mcin.ddata;
            res.sd = mct.rwsd;
            res.kernel = kernel_idx;
            res.blocksz = block_idx;
            res.samples = mcin.Ns;
            res.burnSamples = mcin.burnin;
            res.cpuTime = 0;
            write_perf_out_csv(outdir, res, false);

            free_sample_vectors(mcdata);
            free_autocorrelation_vectors(secv);
            mcin.tune = 0;  // stop tuning after first time
            fprintf(stderr, "Nd: %d, Kernel: %d, block: %d DONE!\n", mcin.Nd, kernel_idx, block_idx);
        }
    }

    mcin.Ns /= 20;      // run cpu for less iterations
    mcin.burnin /= 20;

    malloc_sample_vectors(&mcdata, mcin);
    malloc_autocorrelation_vectors(&secv, sec);

    for(dim_idx=0; dim_idx<mcin.ddata; dim_idx++){ mcdata.burn[dim_idx] = 0; } 
    
    mcin.tune = 0;      // don't perform tuning on cpu. use sd from the gpu

    start  = clock();
    cpu_sampler(data, r, mcin, &mct, mcdata, &res);
    stop = clock() - start;

    res.samplerTime = stop * 1000 / CLOCKS_PER_SEC;  // sampler time in ms
    res.ess = get_ess(mcdata.samples, mcin.Ns, mcin.ddata, sec.lagidx, secv.circ);

    res.device = 1;     // cpu
    res.Nd = mcin.Nd;
    res.dim = mcin.ddata;
    res.sd = mct.rwsd;
    res.kernel = 0;
    res.blocksz = 0;    
    res.samples = mcin.Ns;
    res.burnSamples = mcin.burnin; 
    res.gpuTime = 0;
       
    write_perf_out_csv(outdir, res, false);

    free_sample_vectors(mcdata);
    free_autocorrelation_vectors(secv);
    free_data_vectors(data);
    free_rng(r);

    return 0;
}
