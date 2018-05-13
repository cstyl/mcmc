import numpy as np
import matplotlib.pyplot as plt
import csv

filepath = 'out/cpu/synthetic/shift_autocorrelation.csv'

autocorrelation=[]
lagk=[]

with open(filepath,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        autocorrelation.append(float(row[1]))
        lagk.append(int(row[0]))

data = {'lagk' : lagk,
        'autocorrelation' : autocorrelation}

plt.figure(1)
plt.title('Shift Autocorrelation Plot for Synthetic case')
plt.xlabel('lag k')
plt.scatter('lagk', 'autocorrelation', data=data, marker='.')
plt.ylabel('autocorrelation, ' + r'$\rho$')
# plt.axis([0, 26000, -62, 62])
plt.grid(True)
plt.show()

filepath = 'out/cpu/synthetic/circular_autocorrelation.csv'

autocorrelation=[]
lagk=[]

with open(filepath,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        autocorrelation.append(float(row[1]))
        lagk.append(int(row[0]))

data = {'lagk' : lagk,
        'autocorrelation' : autocorrelation}

plt.figure(2)
plt.title('Circular Autocorrelation Plot for Synthetic case')
plt.xlabel('lag k')
plt.scatter('lagk', 'autocorrelation', data=data, marker='.')
plt.ylabel('autocorrelation, ' + r'$\rho$')
# plt.axis([0, 26000, -62, 62])
plt.grid(True)
plt.show()

filepath = 'out/cpu/synthetic/autocorrelation_lag.csv'

autocorrelation=[]
lagk=[]

with open(filepath,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)
    for row in plots:
        autocorrelation.append(float(row[1]))
        lagk.append(int(row[0]))

data = {'lagk' : lagk,
        'autocorrelation' : autocorrelation}

plt.figure(3)
plt.title('Autocorrelation Plot for Synthetic case')
plt.xlabel('lag k')
plt.scatter('lagk', 'autocorrelation', data=data, marker='.')
plt.ylabel('autocorrelation, ' + r'$\rho$')
# plt.axis([0, 26000, -62, 62])
plt.grid(True)
plt.show()

# filepath = '../../out/cpu/mnist/autocorrelation.csv'

# autocorrelation=[]
# lagk=[]

# with open(filepath,'r') as csvfile:
#     plots = csv.reader(csvfile, delimiter=',')
#     next(plots, None)
#     for row in plots:
#         autocorrelation.append(float(row[1]))
#         lagk.append(int(row[0]))

# data = {'lagk' : lagk,
#         'autocorrelation' : autocorrelation}

# plt.figure(3)
# plt.title('Autocorrelation Plot for Mnist case')
# plt.xlabel('lag k')
# plt.scatter('lagk', 'autocorrelation', data=data, marker='.')
# plt.ylabel('autocorrelation, ' + r'$\rho$')
# # plt.axis([0, 26000, -62, 62])
# plt.grid(True)
# plt.show()

# filepath = '../../out/cpu/mnist/circular_autocorrelation.csv'

# autocorrelation=[]
# lagk=[]

# with open(filepath,'r') as csvfile:
#     plots = csv.reader(csvfile, delimiter=',')
#     next(plots, None)
#     for row in plots:
#         autocorrelation.append(float(row[1]))
#         lagk.append(int(row[0]))

# data = {'lagk' : lagk,
#         'autocorrelation' : autocorrelation}

# plt.figure(4)
# plt.title('Circular Autocorrelation Plot for Mnist case')
# plt.xlabel('lag k')
# plt.scatter('lagk', 'autocorrelation', data=data, marker='.')
# plt.ylabel('autocorrelation, ' + r'$\rho$')
# # plt.axis([0, 26000, -62, 62])
# plt.grid(True)
# plt.show()