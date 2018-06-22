import numpy as np
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process dataset size and dimensionality')
	parser.add_argument('-dir', dest='dir', type=str, help='Give the directory name of the output')
	parser.add_argument('-sz', dest='sz', type=int, help='an integer for dataset size')
	parser.add_argument('-dim', dest='dim', type=int, help='an integer for dimensionality')
	args = parser.parse_args()

	DIR = args.dir
	DATASET = args.sz
	DIM = args.dim 

	mnistDir = "data/mnist_pca.csv"
	path = "data/" + str(DIR) + "/mnist.csv"

	data = np.genfromtxt(mnistDir, dtype= 'float', delimiter = ",")

	extracted_data = data[0:DATASET, 0:DIM+1]

	np.savetxt(path, extracted_data, delimiter=',')