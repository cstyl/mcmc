import numpy as np
import argparse
import csv

def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset size and dimensionality')
    parser.add_argument('-sz', dest='sz', type=int, help='an integer for dataset size')
    parser.add_argument('-dim', dest='dim', type=int, help='an integer for dimensionality')
    args = parser.parse_args()

    DATASET = args.sz
    DIM = args.dim 

    data = np.zeros((DATASET, DIM+1))
    b = np.random.randint(-15, 15, DIM)
    b[0] = 1
    path = "data/synthetic_betas.csv"
    np.savetxt(path, b, fmt = "%d", newline='\n', delimiter=",")
    # b = [-10, 5, 10]

    for j in range(1,DIM+1):
        for i in range(0,DATASET):
            if j==1:
                data[i][j] = 1
            else:
                data[i][j] = np.random.uniform(0,1)

    for i in range(0,DATASET):
        acc = 0;
        for j in range(1,DIM+1):
            acc += data[i][j] * b[j-1]
        if acc<0:
            data[i][0] = -1
        else:
            data[i][0] = 1

    path = "data/synthetic.csv"
    csv_writer(data, path)