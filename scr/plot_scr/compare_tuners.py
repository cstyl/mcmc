import numpy as np
import matplotlib.pyplot as plt

inDir = 'res/compare_tuners/out/'
outDir = 'res/compare_tuners/'

testCase = ['synthetic', 'mnist']

for i in testCase:
	X = np.genfromtxt(inDir + i + '.csv', delimiter=',')

	plt.figure()
	plt.plot(X[:,0], X[:,3]/X[:,2], '-*')

	plt.xlabel('Dimensionality')
	plt.ylabel(r'Ratio of $ESS_e/ESS_t$ per second')
	plt.grid(True)
	plt.savefig(outDir + 'compare_tuners_' + i + '.eps', format='eps', dpi=1000)
	plt.close()