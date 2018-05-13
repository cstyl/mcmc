import numpy as np
import matplotlib.pyplot as plt

filepath = 'out/cpu/synthetic/burn_out.csv'
X = np.genfromtxt(filepath, delimiter=',')
# print(X)

filepath = 'out/cpu/synthetic/raw_out.csv'
Y = np.genfromtxt(filepath, delimiter=',')
# print(Y)
Z = np.concatenate((X,Y), axis=0)
# print(Z)

legend = []
plt.figure(1)
plt.title('Trace Plot for ' + r'$\theta$' + ' from synthetic case')

for i in range(Z.shape[1]):
    # legend.append(r'$\theta_$i$$')
    plt.plot(Z[:,i], label=r'$\theta$'+str(i))

plt.xlabel('Steps')
plt.ylabel(r'$\theta$')
# plt.axis([0, 26000, -62, 62])
plt.legend()
plt.grid(True)
plt.show()