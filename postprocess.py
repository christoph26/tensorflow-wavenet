import preprocess
import h5py
import argparse
import numpy as np
import librosa

sr = 44100

def write_wav(waveform, sample_rate, filename):
	'''
	Save waveform as wav file
	'''
	y = np.array(waveform)
	librosa.output.write_wav(filename, y, sample_rate)
	print('WAV written at {}'.format(filename))

def postprocess(pca_file, coeff_file, wav_file):

	# load the frequencies of the produced file
	freq = preprocess.load_pca(pca_file, coeff_file)

	# convert the frequencies in 1d audio signal
	audio = preprocess.load_freq(freq)

	write_wav(audio, sr, wav_file)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--pca_file", type=str, required=True, help="Path of the pca mean / variance input file")
	parser.add_argument("--coeff_file", type=str, required=True, help="Path of the pca coefficient input file")
	parser.add_argument("--wav_file", type=str, required=True, help="Path of the wav output")
	args = parser.parse_args()

	postprocess(args.pca_file, args.coeff_file, args.wav_file)