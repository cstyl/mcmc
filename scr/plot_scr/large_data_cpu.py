import numpy as np
import matplotlib.pyplot as plt

legend = []

indir = 'res/large_data_cpu/out/'
outdir = 'res/large_data_cpu/'

filepath = indir + 'cpu_500.csv'
X500 = np.genfromtxt(filepath, delimiter=',',  skip_header=1)
filepath = indir + 'cpu_200.csv'
X200 = np.genfromtxt(filepath, delimiter=',',  skip_header=1)
filepath = indir + 'cpu_100.csv'
X100 = np.genfromtxt(filepath, delimiter=',',  skip_header=1)
filepath = indir + 'cpu_50.csv'
X50 = np.genfromtxt(filepath, delimiter=',',  skip_header=1)
filepath = indir + 'cpu_20.csv'
X20 = np.genfromtxt(filepath, delimiter=',',  skip_header=1)
filepath = indir + 'cpu_10.csv'
X10 = np.genfromtxt(filepath, delimiter=',',  skip_header=1)
filepath = indir + 'cpu_3.csv'
X3 = np.genfromtxt(filepath, delimiter=',',  skip_header=1)

plt.figure()
plt.plot(X500[:,1], X500[:,2]/1000, '-*', label='Dim=500')
plt.plot(X200[:,1], X200[:,2]/1000, '-*', label='Dim=200')
plt.plot(X100[:,1], X100[:,2]/1000, '-*', label='Dim=100',)
plt.plot(X50[:,1], X50[:,2]/1000, '-*', label='Dim=50')
plt.plot(X20[:,1], X20[:,2]/1000, '-*', label='Dim=20')
plt.plot(X10[:,1], X10[:,2]/1000, '-*', label='Dim=10')
plt.plot(X3[:,1], X3[:,2]/1000, '-*', label='Dim=3')

plt.xlabel('Datapoints')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)
plt.savefig(outdir + 'samplerTime.eps', format='eps', dpi=1000)
plt.close()

print("Sampler Time Plot for Synthetic Case extracted at " + outdir)

plt.figure()
plt.semilogx(X500[:,1], X500[:,2]/1000, '-*', label='Dim=500')
plt.semilogx(X200[:,1], X200[:,2]/1000, '-*', label='Dim=200')
plt.semilogx(X100[:,1], X100[:,2]/1000, '-*', label='Dim=100',)
plt.semilogx(X50[:,1], X50[:,2]/1000, '-*', label='Dim=50')
plt.semilogx(X20[:,1], X20[:,2]/1000, '-*', label='Dim=20')
plt.semilogx(X10[:,1], X10[:,2]/1000, '-*', label='Dim=10')
plt.semilogx(X3[:,1], X3[:,2]/1000, '-*', label='Dim=3')

plt.xlabel('Datapoints')
plt.ylabel('Time (s)')
plt.legend(loc=2)
plt.grid(True)
plt.savefig(outdir + 'logsamplerTime.eps', format='eps', dpi=1000)
plt.close()

print("Sampler Time semilogxPlot for Synthetic Case extracted at " + outdir)

plt.figure()
plt.plot(X500[:,1], X500[:,3]/1000, '-*', label='Dim=500')
plt.plot(X200[:,1], X200[:,3]/1000, '-*', label='Dim=200')
plt.plot(X100[:,1], X100[:,3]/1000, '-*', label='Dim=100')
plt.plot(X50[:,1], X50[:,3]/1000, '-*', label='Dim=50')
plt.plot(X20[:,1], X20[:,3]/1000, '-*', label='Dim=20')
plt.plot(X10[:,1], X10[:,3]/1000, '-*', label='Dim=10')
plt.plot(X3[:,1], X3[:,3]/1000, '-*', label='Dim=3')

plt.xlabel('Datapoints')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)
plt.savefig(outdir + 'mcmcTime.eps', format='eps', dpi=1000)
plt.close()

print("MCMC Time Plot for Synthetic Case extracted at " + outdir)

plt.figure()
plt.plot(X500[:,1], X500[:,4]/1000, '-*', label='Dim=500')
plt.plot(X200[:,1], X200[:,4]/1000, '-*', label='Dim=200')
plt.plot(X100[:,1], X100[:,4]/1000, '-*', label='Dim=100')
plt.plot(X50[:,1], X50[:,4]/1000, '-*', label='Dim=50')
plt.plot(X20[:,1], X20[:,4]/1000, '-*', label='Dim=20')
plt.plot(X10[:,1], X10[:,4]/1000, '-*', label='Dim=10')
plt.plot(X3[:,1], X3[:,4]/1000, '-*', label='Dim=3')

plt.xlabel('Datapoints')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)
plt.savefig(outdir + 'burnTime.eps', format='eps', dpi=1000)
plt.close()

print("Burn Time Plot for Synthetic Case extracted at " + outdir)