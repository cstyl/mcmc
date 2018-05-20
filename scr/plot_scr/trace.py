import numpy as np
import matplotlib.pyplot as plt

filepath = 'out/cpu/synthetic/burn_out.csv'
Xc = np.genfromtxt(filepath, delimiter=',')

filepath = 'out/cpu/synthetic/raw_out.csv'
Yc = np.genfromtxt(filepath, delimiter=',')
Zc = np.concatenate((Xc,Yc), axis=0)

filepath = 'out/gpu/synthetic/burn_out.csv'
Xg = np.genfromtxt(filepath, delimiter=',')

filepath = 'out/gpu/synthetic/raw_out.csv'
Yg = np.genfromtxt(filepath, delimiter=',')
Zg = np.concatenate((Xg,Yg), axis=0)

legend = []
plt.figure(1)
plt.title('Trace Plot for ' + r'$\theta$' + ' from synthetic case')

for i in range(Zc.shape[1]):
# for i in range(0,5):
    # legend.append(r'$\theta_$i$$')
    plt.plot(Zc[:,i], label=r'$\theta_c$'+str(i))
    plt.plot(Zg[:,i], label=r'$\theta_g$'+str(i))

plt.xlabel('Steps')
plt.ylabel(r'$\theta$')
# plt.axis([0, 26000, -62, 62])
plt.legend()
plt.grid(True)
plt.show()