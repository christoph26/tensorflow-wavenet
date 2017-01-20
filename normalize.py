import numpy as np
import intervaltree
import h5py
import argparse


'''
The function normalize takes as input the pca 
coefficients from the file 'pca_file'
and normalizes them to 0 mean and unit variance.
The resulting coefficients are stored in output_file.
'''
def normalize(pca_file, output_file):

    # open original pca file
    pca_input = h5py.File(pca_file, 'r')
    pca_input = pca_input['coeff']

    pca_data = []

    # for each sample: add the coefficients
    for key in pca_input:
        pca_data += pca_input[key]

    # compute the mean and the variance
    mean = np.mean(pca_data)
    var = np.var(pca_data)

    # subtract the mean and the variance
    pca_data -= mean
    pca_data /= var

    # store the normalized values again
    h5f = h5py.File(output_file, 'w')
    h5f.create_dataset('normalization/mean_', data=mean)
    h5f.create_dataset('normalization/var_', data=var)

    h5f.create_dataset('coeff/normalized', data=pca_data)
    h5f.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_file", type=str, required=True, help="Path of the *.h5 data file with pca results")
	parser.add_argument("--output_file", type=str, default=None, help="Path of the normalized output file")
	args = parser.parse_args()

	normalize(args.data_file, args.output_file)