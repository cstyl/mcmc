import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Process dataset size and dimensionality')
parser.add_argument('-v', dest='v', type=int, help='Choose which implementations to compare')
parser.add_argument('-d', dest='d', type=int, help='Choose between synthetic or mnist')

args = parser.parse_args()

V = args.v
D = args.d

if V==1:
	mainDir = 'res/compare_gpu_cpu'
	caption = 'GPU vs CPU'
	fast = 'gpu_'
	slow = 'cpu_'
elif V==2:
	mainDir = 'res/compare_gpu_sp'
	caption = 'GPU double precision vs GPU single precision'
	fast = 'sp_'
	slow = 'gpu_'
else:
	exit()

if D==1:
	outdir = mainDir + '/'
	indir = mainDir + '/out/'

	dim = ['200', '100', '50', '20', '10']
elif D==2:
	outdir = mainDir + '_mnist/'
	indir = mainDir + '_mnist/out/'

	dim = ['100', '70', '30', '12', '5']
else:
	exit()

filepathg = indir + fast + dim[0] + '.csv'
filepathc = indir + slow + dim[0] + '.csv'
Xg1 = np.genfromtxt(filepathg, delimiter=',')
Xc1 = np.genfromtxt(filepathc, delimiter=',')

filepathg = indir + fast + dim[1] + '.csv'
filepathc = indir + slow + dim[1] + '.csv'
Xg2 = np.genfromtxt(filepathg, delimiter=',')
Xc2 = np.genfromtxt(filepathc, delimiter=',')

filepathg = indir + fast + dim[2] + '.csv'
filepathc = indir + slow + dim[2] + '.csv'
Xg3 = np.genfromtxt(filepathg, delimiter=',')
Xc3 = np.genfromtxt(filepathc, delimiter=',')

filepathg = indir + fast + dim[3] + '.csv'
filepathc = indir + slow + dim[3] + '.csv'
Xg4 = np.genfromtxt(filepathg, delimiter=',')
Xc4 = np.genfromtxt(filepathc, delimiter=',')

filepathg = indir + fast + dim[4] + '.csv'
filepathc = indir + slow + dim[4] + '.csv'
Xg5 = np.genfromtxt(filepathg, delimiter=',')
Xc5 = np.genfromtxt(filepathc, delimiter=',')

#time vs size
plt.figure()
plt.plot(Xc1[:,1], Xg1[:,6]/Xc1[:,6], '-*', label='Dim=' + dim[0])
plt.plot(Xc2[:,1], Xg2[:,6]/Xc2[:,6], '-*', label='Dim=' + dim[1])
plt.plot(Xc3[:,1], Xg3[:,6]/Xc3[:,6], '-*', label='Dim=' + dim[2])
plt.plot(Xc4[:,1], Xg4[:,6]/Xc4[:,6], '-*', label='Dim=' + dim[3])
plt.plot(Xc5[:,1], Xg5[:,6]/Xc5[:,6], '-*', label='Dim=' + dim[4])


plt.xlabel('Size (Bytes)')
plt.ylabel('Speed Up (Times)')
plt.legend()
plt.grid(True)
plt.savefig(outdir + 'speedUpGpuCpu.eps', format='eps', dpi=1000)
plt.close()

print('Speedup Plot for ' + caption + ' extracted in ' + outdir)

plt.figure()
plt.semilogx(Xc1[:,1], Xg1[:,6]/Xc1[:,6], '-*', label='Dim=' + dim[0])
plt.semilogx(Xc2[:,1], Xg2[:,6]/Xc2[:,6], '-*', label='Dim=' + dim[1])
plt.semilogx(Xc3[:,1], Xg3[:,6]/Xc3[:,6], '-*', label='Dim=' + dim[2])
plt.semilogx(Xc4[:,1], Xg4[:,6]/Xc4[:,6], '-*', label='Dim=' + dim[3])
plt.semilogx(Xc5[:,1], Xg5[:,6]/Xc5[:,6], '-*', label='Dim=' + dim[4])

plt.xlabel('Size (Bytes)')
plt.ylabel('Speed Up (Times)')
plt.legend(loc=2)
plt.grid(True)
plt.savefig(outdir + 'logspeedUpGpuCpu.eps', format='eps', dpi=1000)
plt.close()

print('Speedup semilogx Plot for ' + caption + ' extracted in ' + outdir)