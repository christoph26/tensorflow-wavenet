import fnmatch
import os
import re
import threading

import librosa
import h5py
import numpy as np
import tensorflow as tf

import csv

def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, filename


def load_vctk_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the VCTK dataset, and
    additionally the ID of the corresponding speaker.'''
    files = find_files(directory)
    speaker_re = re.compile(r'p([0-9]+)_([0-9]+)\.wav')
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        matches = speaker_re.findall(filename)[0]
        speaker_id, recording_id = [int(id_) for id_ in matches]
        yield audio, speaker_id

def load_npz_audio(directory, sample_rate):
    files = find_files(directory, pattern='*.npz')
    for filename in files:
        keys = []
        data = np.load(open(filename, 'rb'), encoding='bytes')
        if os.path.isfile(filename[:-4]+"_metadata.csv"):
            with open(filename[:-4]+"_metadata.csv", 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row[1] == "Beethoven" and row[2].find("Piano") >= 0:
                        keys.append(row[0])
        else:
            keys = data.files
        for file_i in keys:
            X,Y = data[str(file_i)]
            X = X.astype("float32")
            X = X.reshape(-1,1)
            yield X, '{}_{}'.format(filename, file_i)

def load_pca_audio(directory, sample_rate):
    files = find_files(directory, pattern='*_pca.h5')
    for filename in files:
        if os.path.isfile(filename[:-7]+"_metadata.csv"):
            keys = []
            with open(filename[:-7]+"_metadata.csv", 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row[1] == "Beethoven" and row[2].find("Piano") >= 0:
                        keys.append("id_"+str(row[0]))
        else:
            keys = h5f['coeff']

        h5f = h5py.File(filename, 'r')
        keys = ["id_2322"]
        for file_i in keys:
            X = h5f['coeff/{}'.format(file_i)].value
            yield X, '{}_{}'.format(filename, file_i)


def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=256):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.silence_threshold = silence_threshold
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, 100)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        if not find_files(audio_dir, pattern="*_pca.h5"):
            raise ValueError("No audio files found in '{}'.".format(audio_dir))

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        buffer_ = np.array([])
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_pca_audio(self.audio_dir, self.sample_rate)
            for audio, filename in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                # Silence threshold not needed with pca data, since this is alrady taken into account when producing the pca data
                #if self.silence_threshold is not None:
                    # Remove silence
                #    audio = trim_silence(audio[:, 0], self.silence_threshold)
                #    audio = audio.reshape(-1, 1)
                #    if audio.size == 0:
                #        print("Warning: {} was ignored as it contains only "
                #              "silence. Consider decreasing trim_silence "
                #              "threshold, or adjust volume of the audio."
                #              .format(filename))

                if self.sample_size:
                    # Cut samples into fixed size pieces
                    buffer_ = np.append(buffer_, audio)
                    while len(buffer_) > self.sample_size:
                        piece = np.reshape(buffer_[:self.sample_size], [-1, 1])
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        buffer_ = buffer_[self.sample_size:]
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: audio})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
