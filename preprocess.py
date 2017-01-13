import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import intervaltree
from sklearn.decomposition import IncrementalPCA
import h5py
import argparse
import os
import csv

fs = 44100
crop_freq_th = 150
window_size = 2048  # 2048-sample fourier windows
stride = window_size #samples between windows

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

def save_frequencies(input_file, output_file, filter_piano):
	if output_file is None:
		output_dict = {}
		for key, data in load_musicnet(input_file, filter_piano):
			freq = freq_extraction(data)
			output_dict[key] = freq
			if VERBOSE:
				print("frequencies of file {} extracted ({})".format(key, len(data)))
		if VERBOSE:
			print("frequency extraction done")
		return output_dict
	else:
		h5f = h5py.File(output_file, 'w')
		for key, data in load_npz(input_file):#load_musicnet(input_file):
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

def save_pca_passed_freq(freq_dict, output_file):
	'''
	Input file = frequency file
	'''
	pca = Audio_PCA()
	for key, data in freq_dict.items():
		pca.partial_fit(data)

		if VERBOSE:
			print("pca of file {} fitted".format(key))

	if VERBOSE:
		print("pca fitting done")
	h5f = h5py.File(output_file, 'w')
	h5f.create_dataset('pca/mean_', data=pca.mean_)
	h5f.create_dataset('pca/components_', data=pca.components_)

	pca_dict= {}
	for key, data in freq_dict.items():
		coeff = pca.transform(data)
		pca_dict[key] = coeff
		if VERBOSE:
			print("pca coefficients of file {} generated".format(key))

	if VERBOSE:
		print("pca coefficients generation done")

	# calculate mean an variance
	all_pca = np.array([]).reshape((-1,coeff_per_window))
	for key in pca_dict:
		all_pca = np.concatenate((all_pca, pca_dict[key]), axis=0)
	mean = np.mean(all_pca, axis=0)
	var = np.var(all_pca, axis=0)

	h5f.create_dataset('normalize/mean', data=mean)
	h5f.create_dataset('normalize/var', data=var)

	if VERBOSE:
		print("Calculated mean and variance. Starting normalization.")

	for key in freq_dict:
		pca_dict[key] = pca_dict[key] - mean
		pca_dict[key] = pca_dict[key] / (var / 15.0)
		h5f.create_dataset('coeff/{}'.format(key), data=pca_dict[key])
		print("Saved file " + str(key))

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

def load_pca(input_file, coeff_file=None, output_file=None):
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

	if coeff_file:
		h5f = h5py.File(coeff_file, 'r')
	else:
		h5f = h5f["coeff"]

	for key in h5f:
		coeff = h5f[key].value
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

def load_freq(input, output_file=None, abs_value=False):
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
			if abs_value:
				Xs_red[i * stride:i * stride + window_size] += np.abs(np.fft.ifft(Xs))
			else:
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
	return data

def load_npz(filename):
	data = np.load(open(filename, 'rb'), encoding='bytes')
	if VERBOSE:
		print("reading {} files".format(len(data.files)))
	for key in data.files:
		print("reading file {}".format(key))
		yield key, data[key][0].astype("float32")

def load_musicnet(filename, filter_piano=False):
	if filter_piano:
		#Get all ids with Piano music
		valid_keys = []
		if os.path.isfile(filename[:-3] + "_metadata.csv"):
			with open(filename[:-3] + "_metadata.csv", 'r') as f:
				reader = csv.reader(f)
				for row in reader:
					if row[4].find("Piano") >= 0:
						valid_keys.append("id_"+str(row[0]))
		else:
			print("Metadata file could not be found.")

	f = h5py.File(filename, 'r')
	for key in f:
		if filter_piano:
			if key in valid_keys:
				yield key, f[key]['data'].value
			else:
				print("Skipped key" + key)
		else:
			yield key, f[key]['data'].value

def load_h5f(filename):
	f = h5py.File(filename, 'r')
	for key in f:
		yield key, f[key].value

def preprocess(data_file, freq_file, pca_file, filter_piano):
	if not freq_file:
		freq_file = data_file[:-4]+"_freq.h5"
	if not pca_file:
		pca_file = data_file[:-4]+"_pca.h5"

	freq_dict = save_frequencies(data_file, None, filter_piano)
	save_pca_passed_freq(freq_dict, pca_file)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_file", type=str, required=True, help="Path of the *.npz data file")
	parser.add_argument("--freq_file", type=str, default=None, help="Path of the frequencies output")
	parser.add_argument("--pca_file", type=str, default=None, help="Path of the pca output file")
	parser.add_argument('--filter_piano', type=bool, default=False, help='Should a metadata file be used to filter piano pieces.')
	args = parser.parse_args()

	preprocess(args.data_file, args.freq_file, args.pca_file, args.filter_piano)

	#np.save(output_file, proc_data)
