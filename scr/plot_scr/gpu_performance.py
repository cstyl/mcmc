import numpy as np
import matplotlib.pyplot as plt

indir = 'res/gpu_performance/out/'
outdir = 'res/gpu_performance/'

file = ['200', '100', '50', '20', '10']

for dim in file:
	filepath = indir + 'gpu_' + dim 

	X = np.genfromtxt(filepath  + '.csv', delimiter=',', skip_header=1)

	blocks = X[:,6]

	data64 = X[blocks==64]
	data128 = X[blocks==128]
	data256 = X[blocks==256]
	data512 = X[blocks==512]
	data1024 = X[blocks==1024]

	#time vs size
	plt.figure()
	plt.plot(data64[:,0]*data64[:,1]*8, data64[:,2], '-*', label='Blocks=64')
	plt.plot(data128[:,0]*data128[:,1]*8, data128[:,2], '-*', label='Blocks=128')
	plt.plot(data256[:,0]*data256[:,1]*8, data256[:,2], '-*', label='Blocks=256')
	plt.plot(data512[:,0]*data512[:,1]*8, data512[:,2], '-*', label='Blocks=512')
	plt.plot(data1024[:,0]*data1024[:,1]*8, data1024[:,2], '-*', label='Blocks=1024')


	plt.xlabel('Size (Bytes)')
	plt.ylabel('Time (ms)')
	plt.legend(loc=2)
	plt.grid(True)
	plt.savefig(outdir + 'samplerTime' + dim + '.eps', format='eps', dpi=1000)
	plt.close()

	print("Sampler Time Plot for " + dim + "-dim Synthetic Case extracted at " + outdir)

	# semilogx graph
	plt.figure()
	plt.semilogx(data64[:,0]*data64[:,1]*8, data64[:,2], '-*', label='Blocks=64')
	plt.semilogx(data128[:,0]*data128[:,1]*8, data128[:,2], '-*', label='Blocks=128')
	plt.semilogx(data256[:,0]*data256[:,1]*8, data256[:,2], '-*', label='Blocks=256')
	plt.semilogx(data512[:,0]*data512[:,1]*8, data512[:,2], '-*', label='Blocks=512')
	plt.semilogx(data1024[:,0]*data1024[:,1]*8, data1024[:,2], '-*', label='Blocks=1024')

	plt.xlabel('Size (Bytes)')
	plt.ylabel('Time (ms)')
	plt.legend(loc=2)
	plt.grid(True)
	plt.savefig(outdir + "logsamplerTime" + dim + ".eps", format='eps', dpi=1000)
	plt.close()

	print("Sampler Time semilogx Plot for " + dim + "-dim Synthetic Case extracted at " + outdir)

	# Bandwidth vs size
	plt.figure()
	plt.plot(data64[:,0]*data64[:,1]*8, data64[:,5], '-*', label='Blocks=64')
	plt.plot(data128[:,0]*data128[:,1]*8, data128[:,5], '-*', label='Blocks=128')
	plt.plot(data256[:,0]*data256[:,1]*8, data256[:,5], '-*', label='Blocks=256')
	plt.plot(data512[:,0]*data512[:,1]*8, data512[:,5], '-*', label='Blocks=512')
	plt.plot(data1024[:,0]*data1024[:,1]*8, data1024[:,5], '-*', label='Blocks=1024')

	plt.ylabel('Bandwidth (GB/s)')
	plt.xlabel('Size (Bytes)')
	plt.legend(loc=2)
	plt.grid(True)
	plt.savefig(outdir + 'Bandwidth' + dim + '.eps', format='eps', dpi=1000)
	plt.close()

	print("Bandwidth Plot for " + dim + "-dim Synthetic Case extracted at " + outdir)