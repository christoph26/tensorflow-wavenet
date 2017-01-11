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
window_size = 2048  # 2048-sample fourier windows
stride = window_size        # 512 samples between windows

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
	data = trim_silence(data, silence_th)
	data = data.copy()

	freq_length = int(np.ceil((len(data)-window_size)/stride))+1
	data.resize(int((freq_length-1)*stride+window_size))

	freq = np.empty([freq_length, crop_freq_th], dtype=complex)

	for i in range(freq.shape[0]):
		Xs = fft(data[i*stride:i*stride+window_size])
		freq[i,:] = Xs[:crop_freq_th]

	return freq.flatten()

def calculate_frequencies(input_file):
	output_dict = {}
	for key, data in load_musicnet(input_file):#load_npz(input_file):  # l
		freq = freq_extraction(data)
		freq_real = np.concatenate((freq.real, freq.imag))
		output_dict[key] = freq_real
		if VERBOSE:
			print("frequencies of file {} extracted ({})".format(key, len(data)))
	if VERBOSE:
		print("frequency extraction done")
	return output_dict

def load_freq(generated_input):

	for i in xrange(generated_input.shape[0]/stride):
		Xs = np.zeros(window_size, dtype=complex)
	gen = load_h5f(input) if type(input) == str else dict_to_gen(input)

	if output_file:
		h5f = h5py.File(output_file, 'w')
	else:
		ret = dict()

	for key, freq in gen:
		Xs_red = np.zeros(freq.shape[0] * stride + window_size)
		for i in range(freq.shape[0]):
			Xs = np.zeros(window_size, dtype=complex)
			Xs[:crop_freq_th] = freq[i]
			Xs[-crop_freq_th + 1:] = freq[i, 1:][::-1]
			Xs_red[i * stride:i * stride + window_size] += np.real(np.fft.ifft(Xs))

		if output_file:
			h5f.create_dataset(key, data=Xs_red)
		else:
			ret[key] = Xs_red
		if VERBOSE:
			print("frequencies of file {} converted into audio signal".format(key))


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

def preprocess(data_file, freq_file):
	if not freq_file:
		freq_file = data_file[:-3] + "_frequencies.h5"

	freq_dict = calculate_frequencies(data_file)

	#calculate mean an variance
	all_freqs = []
	for key in freq_dict:
		all_freqs = np.concatenate((all_freqs, freq_dict[key]))
	mean = np.mean(all_freqs)
	var = np.var(all_freqs)

	h5f = h5py.File(freq_file, 'w')
	h5f.create_dataset('normalize/mean', data=mean)
	h5f.create_dataset('normalize/var', data=var)

	if VERBOSE:
		print("Calculated mean and variance. Starting normalization.")

	for key, data in freq_dict:
		freq_dict[key] -= mean
		freq_dict[key] /= var
		h5f.create_dataset('coeff/{}'.format(key), data=freq_dict[key])

	h5f.close()
	if VERBOSE:
		print("Frequency file saved.")

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_file", type=str, required=True, help="Path of the *.npz data file")
	parser.add_argument("--freq_file", type=str, default=None, help="Path of the frequencies output file")
	args = parser.parse_args()

	preprocess(args.data_file, args.freq_file)

	#np.save(output_file, proc_data)
