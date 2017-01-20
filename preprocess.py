import librosa
import numpy as np
from scipy import fft
import intervaltree
from sklearn.decomposition import IncrementalPCA
import h5py
import argparse
import os
import csv

crop_freq_th = 150 # from the originally 2048 frequencies only the first 150 are considered relevant
window_size = 2048  # 2048-sample fourier windows
stride = 512 #samples between windows

coeff_per_window = 100 # the number of pca coefficients considered as relevant

silence_th = 0.2 # the value for which to trim the silence at the beginning and at the end of a piece


VERBOSE = True

def trim_silence(audio, threshold):
	'''
	Removes silence at the beginning and end of a sample.
	'''
	energy = librosa.feature.rmse(audio)
	frames = np.nonzero(energy > threshold)
	indices = librosa.core.frames_to_samples(frames)[1]

	# Note: indices can be an empty array, if the whole audio was silence.
	return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

def freq_extraction(data):
	'''
	Requires the raw audio data (1d-array with amplitudes).
	Returns a 2d array with the frequencies of the audio sample.
	The returned frequencies are cropped to the first 'crop_freq_th' ones
	(ignoring unimportant frequencies)
	'''

	# remove the silence at the beginning and at the end of the piece
	data = trim_silence(data, silence_th)

	data = data.copy()

	# freq_length indicates to how many time steps the audio piece gets reduced 
	# (i.e. how many fourier windows are generated)
	freq_length = int(np.ceil((len(data)-window_size)/stride))+1

	# the data is resized s.t. the audio fits exactly the window_size and the stride
	data.resize(int((freq_length-1)*stride+window_size))

	# initialize empty result variable
	freq = np.empty([freq_length, crop_freq_th], dtype=complex)

	# compute the fft of the fourier window with size window_size for every stride
	for i in range(freq.shape[0]):
		Xs = fft(data[i*stride:i*stride+window_size])
		freq[i,:] = Xs[:crop_freq_th]

	return freq

def save_frequencies(input_file, output_file, filter_piano):
	'''
	Load the music samples in raw audio format from input_file
	and write the extracted frequencies (freq_extraction) to output_file
	if filter_piano is True, only the piano pieces are considered.
	'''
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
	Given an input frequency file, it does PCA on the data and stores the mean,
	the components and the coefficients of the PCA in output_file
	'''

	pca = Audio_PCA()

	# incrementally fit the frequencies of all the audio files
	for key, data in load_h5f(input_file):
		pca.partial_fit(data)

		if VERBOSE:
			print("pca of file {} fitted".format(key))

	if VERBOSE:
		print("pca fitting done")

	# save the pca mean and components
	h5f = h5py.File(output_file, 'w')
	h5f.create_dataset('pca/mean_', data=pca.mean_)
	h5f.create_dataset('pca/components_', data=pca.components_)

	# transform the data according to the fitted pca and save the coefficients
	for key, data in load_h5f(input_file):
		coeff = pca.transform(data)
		h5f.create_dataset('coeff/{}'.format(key), data=coeff)

		if VERBOSE:
			print("pca coefficients of file {} generated".format(key))

	if VERBOSE:
		print("pca coefficients generation done")
	h5f.close()

def save_pca_passed_freq(freq_dict, output_file, normalize=False):
	'''
	Given the frequency data as an object (not as a file), it produces the PCA mean,
	components and coeffcients on the data and stores it in output_file
	if normalize is True, the pca coefficients are normalized.
	'''

	# incrementally fit the frequencies of all the audio files
	pca = Audio_PCA()
	for key, data in freq_dict.items():
		pca.partial_fit(data)

		if VERBOSE:
			print("pca of file {} fitted".format(key))

	if VERBOSE:
		print("pca fitting done")

	# save the pca mean and components
	h5f = h5py.File(output_file, 'w')
	h5f.create_dataset('pca/mean_', data=pca.mean_)
	h5f.create_dataset('pca/components_', data=pca.components_)

	# transform the data according to the fitted pca and save the coefficients
	pca_dict= {}
	for key, data in freq_dict.items():
		coeff = pca.transform(data)
		pca_dict[key] = coeff
		if VERBOSE:
			print("pca coefficients of file {} generated".format(key))

	if VERBOSE:
		print("pca coefficients generation done")

	if normalize:
		# calculate mean an variance and normalize the pca data

		# gather all the data in one matrix
		all_pca = np.array([]).reshape((-1,coeff_per_window))
		for key in pca_dict:
			all_pca = np.concatenate((all_pca, pca_dict[key]), axis=0)

		# calculate mean and variance
		mean = np.mean(all_pca, axis=0)
		var = np.var(all_pca, axis=0)

		# store the mean and variance
		h5f.create_dataset('normalize/mean', data=mean)
		h5f.create_dataset('normalize/var', data=var)

		if VERBOSE:
			print("Calculated mean and variance. Starting normalization.")

		# normalizing the data and storing the normalized coefficients
		for key in freq_dict:
			pca_dict[key] = pca_dict[key] - mean
			pca_dict[key] = pca_dict[key] / (var / 10.0)
			h5f.create_dataset('coeff/{}'.format(key), data=pca_dict[key])
	else:
		# storing the normalized coefficients without doing normalization before
		for key in freq_dict:
			h5f.create_dataset('coeff/{}'.format(key), data=pca_dict[key])

	h5f.close()
	print("Preprocessing done.")


class Audio_PCA(IncrementalPCA):
	'''
	The only difference to a normal IncrementalPCA object is,
	that it handles imaginary numbers in the feature space.
	To convert an array of imaginary numbers to a real numbered array,
	the real parts of the numbers are just concatenated to the imaginary ones.
	'''
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
	Load the file 'input_file' containing the pca coefficients and outputting
	the frequencies (after doing inverse transform of the pca coefficients)
	if an output_file is specified, it is stored in that file.
	Otherwise it is returned as a dict
	If coeff_file is specified the coefficients are provided from this file.
	Otherwise they are assumed to be included in the input_file

	(input = pca coeffs
	output = frequencies)
	'''

	# reading file
	h5f = h5py.File(input_file, 'r')

	# initializing PCA object according to the stored mean and components
	pca = Audio_PCA()
	pca.components_ = h5f["pca/components_"].value
	pca.mean_ = h5f["pca/mean_"].value

	# initialize the output container(file or dict)
	if output_file:
		h5f_freq = h5py.File(output_file, 'w')
	else:
		freqs = dict()

	# define where the coefficients are specified (either coeff_file or in the input_file)
	if coeff_file:
		h5f = h5py.File(coeff_file, 'r')
	else:
		h5f = h5f["coeff"]

	# inverse transform of all the pca data and write it to the output
	for key in h5f:
		coeff = h5f[key].value
		freq = pca.inverse_transform(coeff)
		if output_file:
			h5f_freq.create_dataset(key, data=freq)
		else:
			freqs[key]=freq

	# close hdf5 files
	h5f.close()

	if output_file:
		h5f_freq.close()
	else:
		return freqs

def dict_to_gen(d):
	'''
	helper function which converts a dictionary to a generator with 2 outputs (key and data)
	'''
	for key in d:
		yield key, d[key]

def load_freq(input, output_file=None, abs_value=False):
	'''
	Takes a dict or a filename as input.
	The input are the frequencies (2d matrix).
	It outputs the generated audio (in raw wave format (1d array)).
	If output_file is specified it is stored in that file.
	Otherwise it is returned as an array.

	If abs_value is True, the absolute value of the inverse fourier transform is taken
	otherwise just the real part of the number.
	'''

	# initialize generator for input data
	gen = load_h5f(input) if type(input) == str else dict_to_gen(input)

	# initialize output
	if output_file:
		h5f = h5py.File(output_file, 'w')
	else:
		ret = dict()

	# process all data
	for key, freq in gen:

		# initialize output variable
		Xs_red = np.zeros(freq.shape[0]*stride+window_size)
		for i in range(freq.shape[0]):

			# compute all 2048 fourier coefficients (first 150 given, the remaining ones are zero)
			Xs = np.zeros(window_size, dtype=complex)
			Xs[:crop_freq_th] = freq[i]
			Xs[-crop_freq_th+1:] = freq[i, 1:][::-1]

			# compute the inverse fourier transform and add the computed data to the output array
			if abs_value:
				Xs_red[i * stride:i * stride + window_size] += np.abs(np.fft.ifft(Xs))
			else:
				Xs_red[i*stride:i*stride+window_size] += np.real(np.fft.ifft(Xs))

		# write the output
		if output_file:
			h5f.create_dataset(key, data=Xs_red)
		else:
			ret[key] = Xs_red

		if VERBOSE:
			print("frequencies of file {} converted into audio signal".format(key))

	# close file or return the data
	if output_file:
		h5f.close()
	else:
		return ret



def load_npz(filename):
	'''
	Load data provided in the npz format.
	This function is a generator producing 2 outputs (key and data)
	'''
	data = np.load(open(filename, 'rb'), encoding='bytes')
	if VERBOSE:
		print("reading {} files".format(len(data.files)))
	for key in data.files:
		print("reading file {}".format(key))
		yield key, data[key][0].astype("float32")

def load_musicnet(filename, filter_piano=False):
	'''
	Generator to load the musicnet data
	if filter_piano is True, only the piano pieces are returned
	'''
	if filter_piano:
		#Get all ids with Piano music
		valid_keys = []

		# open the file which is specified in the same name as filename, but with "_metadata.csv" appended
		if os.path.isfile(filename[:-3] + "_metadata.csv"):
			with open(filename[:-3] + "_metadata.csv", 'r') as f:
				reader = csv.reader(f)
				for row in reader:

					# if piano is contained in the description field, its id is appended to valid_keys
					if row[4].find("Piano") >= 0:
						valid_keys.append("id_"+str(row[0]))
		else:
			print("Metadata file could not be found.")

	# read all the data
	f = h5py.File(filename, 'r')
	for key in f:

		# filter for piano music if specified to do so
		# and afterwards return the pair (key, value)
		if filter_piano:
			if key in valid_keys:
				yield key, f[key]['data'].value
			else:
				print("Skipped key" + key)
		else:
			yield key, f[key]['data'].value

def load_h5f(filename):
	'''
	Helper function to convert a hdf5 file into a generator
	producing (key, value) pairs
	'''
	f = h5py.File(filename, 'r')
	for key in f:
		yield key, f[key].value

def preprocess(data_file, freq_file, pca_file, filter_piano):
	'''
	Preprocesses raw audio files.
	Takes as input (data_file) a raw audio file (containing audio pieces with 1d-audio-wave-amplitudes).
	The freq_file specifies where the frequencies should be stored
	The pca_file specifies where the pca coefficients, mean and components should be stored
	If filter_piano is true, then only the piano pieces are considered.
	'''

	# if the files are not provided, they are generated automatically by adding _freq / _pca to the filename
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
