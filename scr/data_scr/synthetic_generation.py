import numpy as np
import argparse
import csv

def csv_writer(data, path, idx, split):
    """
    Write data to a CSV file path
    """
    line_idx = 0
    with open(path, 'a') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            line_idx += 1
            writer.writerow(line)
        
        if(((idx+1)%(SPLIT/10) == 0) and (SPLIT >= 10)):
            print('Batch ' + str(idx+1) + " Completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset size and dimensionality')
    parser.add_argument('-dir', dest='dir', type=str, help='Give the directory name of the output')
    parser.add_argument('-split', dest='split', type=int, help='Decide in how many batches to produce the data')
    parser.add_argument('-sz', dest='sz', type=int, help='an integer for dataset size')
    parser.add_argument('-dim', dest='dim', type=int, help='an integer for dimensionality')
    args = parser.parse_args()

    DIR = args.dir
    DATASET = args.sz
    DIM = args.dim 
    SPLIT = args.split

    b = np.random.randint(-15, 15, DIM)
    b[0] = 1
    path = "data/" + str(DIR) + "/synthetic_betas.csv"
    np.savetxt(path, b, fmt = "%d", newline='\n', delimiter=",")
    # b = [-10, 5, 10]

    path = "data/" + str(DIR) + "/synthetic.csv"

    block_start_idx = 0;
    block_end_idx = (int)(DATASET/SPLIT)

    for idx in range(0,SPLIT):
        data = np.zeros((block_end_idx, DIM+1))
        
        for j in range(1,DIM+1):
            for i in range(block_start_idx, block_end_idx):
                if j==1:
                    data[i][j] = 1
                else:
                    data[i][j] = np.random.uniform(0,1)
        if(((idx+1)%(SPLIT/10) == 0) and (SPLIT >= 10)):
            print('Batch ' + str(idx+1) + '/' + str(SPLIT) + ': Points Generated')

        for i in range(block_start_idx, block_end_idx):
            acc = 0;
            for j in range(1,DIM+1):
                acc += data[i][j] * b[j-1]
            if acc<0:
                data[i][0] = -1
            else:
                data[i][0] = 1

        if(((idx+1)%(SPLIT/10) == 0) and (SPLIT >= 10)):
            print('Batch ' + str((idx+1)) + '/' + str(SPLIT) + ': Labels Generated')
        
        csv_writer(data, path, idx, SPLIT)