import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Process dataset size and dimensionality')
parser.add_argument('-d', dest='d', type=int, help='Choose between synthetic or mnist')

args = parser.parse_args()

D = args.d

indir = 'res/tunning_cpu/out/'
outdir = 'res/tunning_cpu/'
version = ['_tunned_target_', '_large_sd_', '_small_sd_', '_tune_ess_']

if D==1:
	testCase = 'synthetic'
elif D==2:
	testCase = 'mnist'

for i in version:
	filedir = indir + testCase + i
	plotdir = outdir + testCase + i
	filepath = filedir + 'autocorrelation.csv'
	Auto = np.genfromtxt(filepath, delimiter=',')
	lagk = Auto[:,0]
	autocorrelation = Auto[:,1]

	plt.figure()
	markerline, stemlines, baseline = plt.stem(lagk, autocorrelation, '-')
	plt.xlabel('lagk')
	plt.ylabel('autocorrelation')
	plt.grid(True)
	plt.savefig(plotdir + 'autocorrelation.eps', format='eps', dpi=1000)
	plt.close()

	print("Autocorrelation Plot for " + testCase + "extracted at " + outdir)

	filepath = filedir + 'burn_samples.csv'
	Burn = np.genfromtxt(filepath, delimiter=',')
	filepath = filedir + 'samples.csv'
	PostBurn = np.genfromtxt(filepath, delimiter=',')
	bias_mean = np.mean(PostBurn[:,0])
	Samples = np.concatenate((Burn,PostBurn), axis=0)/ bias_mean
	if D==1:
		Samples *= (-10) 
		PostBurn *= (-10)

	legend = []
	plt.figure()
	plt.plot(Samples[:,1], label=r'$\theta_1$')
	plt.plot(Samples[:,2], label=r'$\theta_2$')
	plt.xlabel('Steps')
	plt.ylabel(r'$\theta_i$' + 'value')
	plt.legend()
	plt.grid(True)
	plt.savefig(plotdir + 'trace.eps', format='eps', dpi=1000)
	plt.close()

	print("Trace Plot for " + testCase + " extracted at " + outdir)

	plt.figure()
	plt.plot(Samples[:,1], Samples[:,2], '-+')
	plt.xlabel(r'$\theta_1$')
	plt.ylabel(r'$\theta_2$')
	plt.grid(True)
	plt.savefig(plotdir + '2Dtrace.eps', format='eps', dpi=1000)
	plt.close()

	print("2D Trace Plot for " + testCase + " extracted at " + outdir)

	n1, bins1, patches1 = plt.hist(PostBurn[:,1], bins=250, alpha=0.75, label=r'$\theta_1$')
	n2, bins2, patches2 = plt.hist(PostBurn[:,2], bins=250, alpha=0.75, label=r'$\theta_2$')
	n1 = np.append(n1,0)
	n2 = np.append(n2,0)

	plt.figure()
	plt.plot(bins1, n1, '-+', label=r'$\theta_1$')
	plt.plot(bins2, n2, '-*', label=r'$\theta_2$')
	plt.xlabel('Value')
	plt.ylabel('Frequency')
	plt.legend()
	plt.grid(True)
	plt.savefig(plotdir + 'hist.eps', format='eps', dpi=1000)
	plt.close()

	print("Histogram Plot for " + testCase + " extracted at " + outdir)