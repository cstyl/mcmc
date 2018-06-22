Accelerating MCMC on GPU
========================

Markov chain Monte Carlo (MCMC) methods have been the main tool for Bayesian Inference. This is due to their ability to generate random samples from probability distributions over many variables. However, for real-world problems, these methods become prohibitively expensive. The evaluation of a costly likelihood for each data point at every iteration, sets a computational bottleneck when it comes to models with large and complex data sets. One way to accelerate the likelihood function is to compute it approximately, either by sub-sampling data or custom precision implementations. Another way is to map the algorithm on parallel devices, such as multi-core CPUs, GPUs and FPGAs, exploiting the strengths and features of each architecture. This project aims to accelerate the MCMC Metropolis-Hastings algorithm on GPU. Also, to explore the advantages and disadvantages of mapping this algorithm on a static architecture, like GPU and the impact of various size datasets. 

The project includes:
- A baseline double-precision (DP) CPU version of the algorithm. 
- An accelerated double-precision GPU version achieving speed ups up to x68 (compared to CPU)
- An accelerated single-precision (SP) GPU version achieving speed ups x2 (compared to DP GPU)
- An adaptation of the unbiased [Mixed Precision MCMC](http://cas.ee.ic.ac.uk/people/ccb98/papers/LiuFPT2015.pdf), previously mapped on FPGA, is implemented on GPU (in progress)

Coming Soon
===========
- Mixed Precision MCMC (working version)
- User Guide on how to build and execute