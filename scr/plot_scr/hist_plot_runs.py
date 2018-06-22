import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse

parser = argparse.ArgumentParser(description='Process dataset size and dimensionality')
parser.add_argument('-d', dest='d', type=int, help='Choose between synthetic or mnist')

args = parser.parse_args()

D = args.d
if D==1:
	# plot the generated normalised samples histogram
	filepath = 'res/runs/out/cpu.csv'
	Xc = np.genfromtxt(filepath, delimiter=',')*(-10)
	filepath = 'res/runs/out/gpu.csv'
	Xg = np.genfromtxt(filepath, delimiter=',')*(-10)
	filepath = 'res/runs/out/sp.csv'
	Xs = np.genfromtxt(filepath, delimiter=',')*(-10)
	outdir = 'res/runs/'
	caseData = 'Synthetic'
elif D==2:
	# plot the generated normalised samples histogram
	filepath = 'res/runs_mnist/out/cpu.csv'
	Xc = np.genfromtxt(filepath, delimiter=',')
	filepath = 'res/runs_mnist/out/gpu.csv'
	Xg = np.genfromtxt(filepath, delimiter=',')
	filepath = 'res/runs_mnist/out/sp.csv'
	Xs = np.genfromtxt(filepath, delimiter=',')	
	outdir = 'res/runs_mnist/'
	caseData = 'MNIST'

legend = []

nc, binsc, patchesc = plt.hist(Xc[:,1], bins=350, alpha=0.75, label=r'CPU MCMC')
ng, binsg, patchesg = plt.hist(Xg[:,1], bins=350, alpha=0.75, label=r'GPU MCMC')
ns, binss, patchess = plt.hist(Xs[:,1], bins=350, alpha=0.75, label=r'GPU SP-MCMC')


nc = np.append(nc,0)
ng = np.append(ng,0)
ns = np.append(ns,0)

plt.figure()
plt.plot(binsc, nc, '-+', label=r'CPU MCMC')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig(outdir + 'cpu_synthetic.eps', format='eps', dpi=1000)

plt.figure()
plt.plot(binsg, ng, '-o', label=r'GPU MCMC')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig(outdir + 'gpu_synthetic.eps', format='eps', dpi=1000)

plt.figure()
plt.plot(binsc, nc, '-+', label=r'CPU MCMC')
plt.plot(binss, ns, '-o', label=r'GPU MCMC')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig(outdir + 'cpu_gpu_synthetic.eps', format='eps', dpi=1000)

plt.figure()
plt.plot(binss, ns, '-o', label=r'GPU SP-MCMC')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig(outdir + 'sp_synthetic.eps', format='eps', dpi=1000)

plt.figure()
plt.plot(binsc, nc, '-+', label=r'CPU MCMC')
plt.plot(binsg, ng, '-o', label=r'GPU MCMC')
plt.plot(binss, ns, '-*', label=r'GPU SP-MCMC')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig(outdir + 'all_synthetic.eps', format='eps', dpi=1000)

print("Multiple Runs plots for " + caseData + " case extracted in " + outdir)