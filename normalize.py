import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import intervaltree
from sklearn.decomposition import IncrementalPCA
import h5py
import argparse

fs = 44100
crop_freq_th = 150
window_size = 2048
window_size = 2048  # 2048-sample fourier windows
stride = 512        # 512 samples between windows

coeff_per_window = 100

silence_th = 0.2

wps = fs/float(stride) # ~86 windows/second (with a fs of 16000)

VERBOSE = True

def normalize(pca_file, output_file):
    if not output_file:
        output_file = pca_file[:-3] + "_pca.h5"

    pca_input = h5py.File(pca_file, 'r')
    pca_input = pca_input['coeff']

    pca_data = []
    for key in pca_input:
        pca_data += pca_input[key]

    #pca_input = h5py.File(pca_file, 'r')['coeff'].value
    mean = np.mean(pca_data)
    var = np.var(pca_data)
    pca_data -= mean
    pca_data /= var

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

	#np.save(output_file, proc_data)
