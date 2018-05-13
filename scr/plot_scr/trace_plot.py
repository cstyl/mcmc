import numpy as np
import matplotlib.pyplot as plt
import csv

# plot the generated samples histogram
filepath = 'out/cpu/synthetic/burn_out.csv'
# x,y,z = np.loadtxt('../../out/raw/synthetic_raw_out.csv', delimiter=',', unpack=True)
x0=[]
x1=[]
x2=[]

# with open(filepath,'r') as csvfile:
#     plots = csv.reader(csvfile, delimiter=',')
#     for row in plots:
#         x0.append(float(row[0]))
#         x1.append(float(row[1]))
#         x2.append(float(row[2]))

filepath = 'out/cpu/synthetic/raw_out.csv'

with open(filepath,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x0.append(float(row[0]))
        x1.append(float(row[1]))
        x2.append(float(row[2]))

i = [i for i in range(len(x0))]

legend = [r'$\theta_0$', r'$\theta_1$', r'$\theta_2$']
plt.figure(1)
plt.title('Trace Plot for ' + r'$\theta$' + ' from synthetic case')
plt.xlabel('Iteration ' + r'$i$')
plt.plot(i,x0, c="g", linewidth ="0.4")
plt.plot(i,x1, c="b", linewidth ="0.4")
plt.plot(i,x2, c="r", linewidth ="0.4")
plt.ylabel(r'$\theta_i$')
plt.axis([0, 26000, -62, 62])
plt.legend(legend)
plt.grid(True)
plt.show()

plt.figure(2)
plt.title('Trace Plot for ' + r'$\theta_0$' + ' from synthetic case')
plt.xlabel('Iteration ' + r'$i$')
plt.plot(i,x0, c="g", linewidth ="0.4")
plt.ylabel(r'$\theta_0$')
# plt.axis([0, 26000, -62, 62])
plt.grid(True)
plt.show()

plt.figure(3)
plt.title('Trace Plot for ' + r'$\theta_1$' + ' from synthetic case')
plt.xlabel('Iteration ' + r'$i$')
plt.plot(i,x1, c="b", linewidth ="0.4")
plt.ylabel(r'$\theta_1$')
# plt.axis([0, 26000, -62, 62])
plt.grid(True)
plt.show()

plt.figure(4)
plt.title('Trace Plot for ' + r'$\theta_2$' + ' from synthetic case')
plt.xlabel('Iteration ' + r'$i$')
plt.plot(i,x2, c="r", linewidth ="0.4")
plt.ylabel(r'$\theta_2$')
# plt.axis([0, 26000, -62, 62])
plt.grid(True)
plt.show()
