import numpy as np
import matplotlib.pyplot as plt
import csv

# plot the generated normalised samples histogram
filepath = 'out/cpu/synthetic/norm_raw_out.csv'
Xc = np.genfromtxt(filepath, delimiter=',')
filepath = 'out/gpu/synthetic/norm_raw_out.csv'
Xg = np.genfromtxt(filepath, delimiter=',')

legend = []
for i in range(Xc.shape[1]):
    plt.figure(i)
    plt.title('Normalised Histogram for ' + r'$\theta$' + str(i) + ' from synthetic case')
    plt.hist(Xc[:,i], bins=50, alpha=0.75, label=r'$\theta_c$'+str(i))
    plt.hist(Xg[:,i], bins=50, alpha=0.75, label=r'$\theta_g$'+str(i))
    plt.xlabel('Value')
    plt.ylabel(r'$Frequency$')
    plt.legend()
    plt.grid(True)
    plt.show()


# histogram for all parameters together
# plt.figure(1)
# plt.title('Histogram for all ' + r'$\theta_s$' + ' from synthetic case')
# legend = [r'$\theta_0$', r'$\theta_1$', r'$\theta_2$']
# plt.hist(x0, 100, normed=0, facecolor='g', alpha=0.75)
# plt.hist(x1, 100, normed=0, facecolor='b', alpha=0.75)
# plt.hist(x2, 100, normed=0, facecolor='r', alpha=0.75)
# plt.legend(legend)
# plt.xlabel(r'$\theta$')
# plt.ylabel('Frequency')
# plt.axis([-15, 10, 0, 600])
# plt.grid(True)
# plt.show()

# histogram for each parameter
# plt.figure(2)
# plt.title('Histogram for ' + r'$\theta_0$' + ' from synthetic case')
# plt.xlabel(r'$\theta_0$')
# plt.text(0.25, 400, r'$\mu_0=$' + str(mu_x0))
# plt.hist(x0, 100, normed=1, facecolor='g', alpha=0.75)
# plt.ylabel('Frequency')
# # plt.axis([0, 2, 0, 600])
# plt.grid(True)
# plt.show()

# plt.figure(3)
# plt.title('Histogram for ' + r'$\theta_1$' + ' from synthetic case')
# plt.xlabel(r'$\theta_1$')
# plt.hist(x1, 100, normed=1, facecolor='b', alpha=0.75)
# plt.ylabel('Frequency')
# # plt.axis([4, 6, 0, 600])
# plt.grid(True)
# plt.show()

# plt.figure(4)
# plt.title('Histogram for ' + r'$\theta_2$' + ' from synthetic case')
# plt.xlabel(r'$\theta_2$')
# plt.hist(x2, 100, normed=1, facecolor='r', alpha=0.75)
# plt.ylabel('Frequency')
# # plt.axis([-12, -8, 0, 600])
# plt.grid(True)
# plt.show()