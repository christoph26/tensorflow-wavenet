# Applying WaveNet on Frequency Data

## Requirements

TensorFlow needs to be installed before running the training script.
TensorFlow 0.10 and the current `master` version are supported.

In addition, [librosa](https://github.com/librosa/librosa) must be installed for reading and writing audio.

To install the required python packages (except TensorFlow), run
```bash
pip install -r requirements.txt
```

## Data source

We used the musicnet which can be downloaded here https://homes.cs.washington.edu/~thickstn/start.html (around 7.1GB).
Download the `musicnet.h5` file and the `musicnet_metadata.csv` file and place it in the same folder (on the same level).

## Preprocessing

To extract the PCA coefficients (to train the network with afterwards), run
```bash
python preprocess.py --data_file=musicnet.h5 --pca_file=corpus/musicnet.h5
```

where `musicnet.h5` is a the data source downloaded from the website above, `corpus/musicnet_pca.h5` the path of the preprocessed pca file (output). By adding `--filter_piano=True`, it can be specified, that only piano pieces should be included

## Training the network

We used the [musicnet] corpus.

In order to train the network, execute
```bash
python train.py --data_dir=corpus
```
to train the network, where `corpus` is a directory containing the preprocessed (pca) `.h5` files.
The script will recursively collect all `.h5` files in the directory.

You can find the configuration of the model parameters in [`wavenet_params.json`](./wavenet_params.json).
These need to stay the same between training and generation.

## Generating audio

You can use the `generate.py` script to generate pca coefficients using a previously trained model.

Run
```
python generate.py --samples 1000 --pca_out_path=generated.h5 model.ckpt-1000
```
where `model.ckpt-1000` needs to be a previously saved model.
You can find these in the `logdir`.
The `--samples` parameter specifies how many pca samples you would like to generate.
The `--pca_out_path` parameter specifies where to store the generated pca file.