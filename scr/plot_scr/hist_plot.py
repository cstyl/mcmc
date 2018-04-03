import numpy as np
import matplotlib.pyplot as plt
import csv

# plot the generated samples histogram
filepath = '../../out/cpu/synthetic/norm_raw_out.csv'
# x,y,z = np.loadtxt('../../out/raw/synthetic_raw_out.csv', delimiter=',', unpack=True)
x0=[]
x1=[]
x2=[]

with open(filepath,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x0.append(float(row[0]))
        x1.append(float(row[1]))
        x2.append(float(row[2]))

mu_x0 = np.mean(x0)
mu_x1 = np.mean(x1)
mu_x2 = np.mean(x2)

print(mu_x0)
print(mu_x1)
print(mu_x2)

# histogram for all parameters together
plt.figure(1)
plt.title('Histogram for all ' + r'$\theta_s$' + ' from synthetic case')
legend = [r'$\theta_0$', r'$\theta_1$', r'$\theta_2$']
plt.hist(x0, 100, normed=0, facecolor='g', alpha=0.75)
plt.hist(x1, 100, normed=0, facecolor='b', alpha=0.75)
plt.hist(x2, 100, normed=0, facecolor='r', alpha=0.75)
plt.legend(legend)
plt.xlabel(r'$\theta$')
plt.ylabel('Frequency')
plt.axis([-15, 10, 0, 600])
plt.grid(True)
plt.show()

# histogram for each parameter
plt.figure(2)
plt.title('Histogram for ' + r'$\theta_0$' + ' from synthetic case')
plt.xlabel(r'$\theta_0$')
plt.text(0.25, 400, r'$\mu_0=$' + str(mu_x0))
plt.hist(x0, 100, normed=0, facecolor='g', alpha=0.75)
plt.ylabel('Frequency')
plt.axis([0, 2, 0, 600])
plt.grid(True)
plt.show()

plt.figure(3)
plt.title('Histogram for ' + r'$\theta_1$' + ' from synthetic case')
plt.xlabel(r'$\theta_1$')
plt.hist(x1, 100, normed=0, facecolor='b', alpha=0.75)
plt.ylabel('Frequency')
plt.axis([4, 6, 0, 600])
plt.grid(True)
plt.show()

plt.figure(4)
plt.title('Histogram for ' + r'$\theta_2$' + ' from synthetic case')
plt.xlabel(r'$\theta_2$')
plt.hist(x2, 100, normed=0, facecolor='r', alpha=0.75)
plt.ylabel('Frequency')
plt.axis([-12, -8, 0, 600])
plt.grid(True)
plt.show()