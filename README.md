# Normalizing flows for jet topic modelling

Paper: M. J. Dolan and A. Ore, _TopicFlow: Disentangling quark and gluon jets with normalizing flows_,
[arXiv:2211.16053 [hep-ph]](https://arxiv.org/abs/2211.16053)

## Dependencies
The code has been tested with the following package versions:
- python 3.9
- tensorflow 2.10.1
- tensorflow-probability 0.18.0
- numpy 1.23.0
- scikit-learn 1.1.2
- scipy 1.8.1
- energyflow 1.3.2

## Usage

### Producing EFP datasets
- First download the EnergyFlow Quark/Gluon dataset, either using [`energyflow.qg_jets.load`](https://energyflow.network/docs/datasets/#quark-and-gluon-jets), or from [Zenodo](https://zenodo.org/record/3164691).
- The script `write-efps.py` can then be used to calculate EFP sets from the data:
```[bash]
$ ./write-efps.py [--max_degree 3] [--paths /path/to/dataset/*.npz]
```
- It may take a long time to convert the entire dataset, so it is recommended to parallelize across files if possible.

### Training a flow model
- The script `train-flow.py` trains quark and gluon jet topic flows on mixed datasets with given quark fractions:
```[bash]
$ ./train-flow.py --purities 0.3 0.7 --savedir outputs
```
- The default settings run the FFJORD model. The spline architecture used for generative classification in the paper can be run with:
```[bash]
$ ./train-flow.py --purities 0.3 0.7 --arch rqs --num_transform_layers 16 --num_residual_blocks 1 --hidden_dim 128 --learning_rate 1e-4 --warmup_epochs 5 --dropout -0.2
```
- The script produces the following outputs:
  - `model_config.p`: Pickled dictionary containing the flow config.
  - `qweights`, `gweights`: Network weights for each flow, which can be restored with `topicflow.utils.load_topic_flows`.
  - `wassersteins.p`: Pickled Wasserstein distances to truth distributions for each EFP, stored as `[quark_dists, gluon_dists]`.
  - `metrics.p`: Pickled generative classification metrics, stored as `{'acc': acc, 'auc': auc}`.
  - `summary`: Tensorboard summary directory.



### Train a classifier

- A simple fully-connected classifier can be trained on mixed datasets using `train-dnn.py`:
```[bash]
$ ./train-dnn.py --purities 0.3 0.7 --savedir outputs
```
- The script produces the following outputs:
  - `weights`: Network weights for the model, which can be restored with `tensorflow.keras.Model.load_weights`.
  - `metrics.p`: Pickled generative classification metrics, stored as [loss, auc, acc].
  - `summary`: Tensorboard summary directory.
