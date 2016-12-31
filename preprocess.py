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

def trim_silence(audio, threshold):
	'''Removes silence at the beginning and end of a sample.'''
	energy = librosa.feature.rmse(audio)
	frames = np.nonzero(energy > threshold)
	indices = librosa.core.frames_to_samples(frames)[1]

	# Note: indices can be an empty array, if the whole audio was silence.
	return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

def freq_extraction(data):
	# coeffs/second = coeff_per_window * wps
	data = trim_silence(data, silence_th)

	data = data.copy()

	freq_length = int(np.ceil((len(data)-window_size)/stride))+1
	data.resize(int((freq_length-1)*stride+window_size))

	freq = np.empty([freq_length, crop_freq_th], dtype=complex)

	for i in range(freq.shape[0]):
		Xs = fft(data[i*stride:i*stride+window_size])
		freq[i,:] = Xs[:crop_freq_th]

	return freq

def save_frequencies(input_file, output_file):
	h5f = h5py.File(output_file, 'w')
	for key, data in load_musicnet(input_file):
		freq = freq_extraction(data)
		h5f.create_dataset(key, data=freq)
		if VERBOSE:
			print("frequencies of file {} extracted ({})".format(key, len(data)))
	h5f.close()
	if VERBOSE:
		print("frequency extraction done")

def save_pca(input_file, output_file):
	'''
	Input file = frequency file
	'''
	pca = Audio_PCA()
	for key, data in load_h5f(input_file):
		pca.partial_fit(data)

		if VERBOSE:
			print("pca of file {} fitted".format(key))

	if VERBOSE:
		print("pca fitting done")
	h5f = h5py.File(output_file, 'w')
	h5f.create_dataset('pca/mean_', data=pca.mean_)
	h5f.create_dataset('pca/components_', data=pca.components_)

	for key, data in load_h5f(input_file):
		coeff = pca.transform(data)
		h5f.create_dataset('coeff/{}'.format(key), data=coeff)

		if VERBOSE:
			print("pca coefficients of file {} generated".format(key))

	if VERBOSE:
		print("pca coefficients generation done")
	h5f.close()

class Audio_PCA(IncrementalPCA):
	def __init__(self):
		super(Audio_PCA, self).__init__(n_components=coeff_per_window)
	def partial_fit(self, freq):
		freq_coeff = np.hstack((freq.real, freq.imag))
		super(Audio_PCA, self).partial_fit(freq_coeff)
	def transform(self, freq):
		freq_coeff = np.hstack((freq.real, freq.imag))
		return super(Audio_PCA, self).transform(freq_coeff)
	def inverse_transform(self, pca_coeff):
		freq_coeff = super(Audio_PCA, self).inverse_transform(pca_coeff)
		nb_freq = int(freq_coeff.shape[1]/2)
		freq = freq_coeff[:,:nb_freq]+1j*freq_coeff[:,nb_freq:]
		return freq

def load_pca(input_file, output_file=None):
	'''
	input = pca coeffs
	output = frequencies
	'''
	h5f = h5py.File(input_file, 'r')
	pca = Audio_PCA()
	pca.components_ = h5f["pca/components_"].value
	pca.mean_ = h5f["pca/mean_"].value

	if output_file:
		h5f_freq = h5py.File(output_file, 'w')
	else:
		freqs = dict()

	for key in h5f["coeff"]:
		coeff = h5f["coeff"][key].value
		freq = pca.inverse_transform(coeff)
		if output_file:
			h5f_freq.create_dataset(key, data=freq)
		else:
			freqs[key]=freq

	h5f.close()

	if output_file:
		h5f_freq.close()
	else:
		return freqs

def dict_to_gen(d):
	for key in d:
		yield key, d[key]

def load_freq(input, output_file=None):
	gen = load_h5f(input) if type(input) == str else dict_to_gen(input)

	if output_file:
		h5f = h5py.File(output_file, 'w')
	else:
		ret = dict()

	for key, freq in gen:
		Xs_red = np.zeros(freq.shape[0]*stride+window_size)
		for i in range(freq.shape[0]):
			Xs = np.zeros(window_size, dtype=complex)
			Xs[:crop_freq_th] = freq[i]
			Xs[-crop_freq_th+1:] = freq[i, 1:][::-1]
			Xs_red[i*stride:i*stride+window_size] += np.real(np.fft.ifft(Xs))

		if output_file:
			h5f.create_dataset(key, data=Xs_red)
		else:
			ret[key] = Xs_red
		if VERBOSE:
			print("frequencies of file {} converted into audio signal".format(key))

	if output_file:
		h5f.close()
	else:
		return ret


def pca_incremental_fit(freq):
	    

	import ipdb; ipdb.set_trace()

	return data

def load_npz(filename):
	data = np.load(open(filename, 'rb'), encoding='bytes')
	if VERBOSE:
		print("reading {} files".format(len(data.files)))
	for key in data.files:
		print("reading file {}".format(key))
		yield key, data[key][0].astype("float32")

def load_musicnet(filename):
	f = h5py.File(filename, 'r')
	for key in f:
		yield key, f[key]['data'].value


def load_h5f(filename):
	f = h5py.File(filename, 'r')
	for key in f:
		yield key, f[key].value

def preprocess(data_file, freq_file=None, pca_file=None):
	if not freq_file:
		freq_file = data_file[:-4]+"_freq.h5"
	if not pca_file:
		pca_file = data_file[:-4]+"_pca.h5"

	#save_frequencies(data_file, freq_file)
	save_pca(freq_file, pca_file)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_file", type=str, required=True, help="Path of the *.npz data file")
	parser.add_argument("--freq_file", type=str, default=None, help="Path of the frequencies output")
	parser.add_argument("--pca_file", type=str, default=None, help="Path of the pca output file")
	args = parser.parse_args()

	#h5f = h5py.File(output_file, 'w')
	#for key, data in load_npz(args.data_file):
	#	processed_data = preprocess(data)
	#	h5f.create_dataset(key, data=processed_data)
	#h5f.close()

	preprocess(args.data_file, args.freq_file, args.pca_file)

	#np.save(output_file, proc_data)
