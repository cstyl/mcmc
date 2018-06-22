import numpy as np
from sklearn.decomposition import PCA
import argparse
import csv

def convert(imgf, labelf, outf, n):
	f = open(imgf, "rb")
	o = open(outf, "w")
	l = open(labelf, "rb")

	f.read(16)
	l.read(8)
	images = []

	for i in range(n):
		image = [ord(l.read(1))]
		for j in range(28*28):
			image.append(ord(f.read(1)))
		images.append(image)

	for image in images:
		o.write(",".join(str(pix) for pix in image)+"\n")

	f.close()
	o.close()
	l.close()

if __name__ == "__main__":
	imageDir = "scr/data_scr/train-images.idx3-ubyte"
	labelDir = "scr/data_scr/train-labels.idx1-ubyte"
	mnistDir = "data/mnist_train.csv"

	print "Decompressing the dataset...",
	convert(imageDir, labelDir, mnistDir, 60000)
	print " Done"

	print "Reading csv input.",
	data = np.genfromtxt(mnistDir, dtype= 'int', delimiter = ",")
	print " Done"


	labels = data[:,0]

	print "Extracting datapoints with labels 7 or 9...",
	features_7 = data[labels==7]
	features_9 = data[labels==9]

	subset = np.concatenate((features_7,features_9), axis=0)
	sublabels = subset[:,0]

	# convert labels
	sublabels[sublabels==7] = -1
	sublabels[sublabels==9] = +1

	subset = np.delete(subset, 0, 1)
	subset = subset.astype(float)
	print " Done"

	# normalize subset
	max_element = np.amax(subset)
	subset = subset / max_element

	dim = subset.shape

	print "Obtaining PCA values...",
	# perform pca
	pca = PCA(n_components=dim[1])
	pca_subset = pca.fit_transform(subset)
	# max_element = np.amax(pca_subset)
	# pca_subset = pca_subset / max_element
	# add the bias term
	pca_subset = np.insert(pca_subset, 0, 1, axis=1)
	max_element = np.amax(pca_subset)
	pca_subset /= max_element

	# add the labels of each point to the left
	pca_dataset = np.insert(pca_subset, 0, sublabels, axis=1)
	print " Done"

	# shuffle rows
	print "Shuffling Dataset...",
	np.random.shuffle(pca_dataset)
	print "Done"

	# extract csv
	print "Extracting csv...",
	outfile = "data/mnist_pca.csv"
	np.savetxt(outfile, pca_dataset, delimiter=',')
	print "Done"