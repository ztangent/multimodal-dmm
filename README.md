# Multimodal Deep Markov Models

A PyTorch implementation of the Multimodal Deep Markov Model (MDMM) and associated inference methods described in [Factorized Inference in Deep Markov Models for Incomplete Multimodal Time Series](https://arxiv.org/abs/1905.13570). Please cite this paper if you use or modify any of this code.

Generalizes the Multimodal Variational Auto-Encoder (MVAE) by [Wu & Goodman](https://papers.nips.cc/paper/7801-multimodal-generative-models-for-scalable-weakly-supervised-learning) and the Deep Markov Model by [Krishnan et al](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14215).

## Setup

After creating a virtual environment with `virtualenv` or `conda`, one can simply install the dependencies in `requirements.txt`. Compatible with both Python 2.7 and Python 3.

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Alternatively, one can install the following packages directly through `pip`:
```
# For basic functionality
pip install torch pandas pyyaml matplotlib

# To download and pre-process the Weizmann video dataset
sudo apt-get install ffmpeg
pip install scipy scikit-video scikit-image requests tqdm opencv-python

# To run the experiment scripts using Ray Tune
pip install ray psutil
```

## Datasets

Before training, the datasets need to be generated or downloaded.

To generate the Spirals dataset, make `datasets` the current directory, then run `python spirals.py`. For a list of options, run `python spirals.py -h`.  

To automatically download and preprocess the Weizmann video dataset of human actions, again make sure that `datasets` is the current directory, then run `python weizmann.py`.

If automated download fails, create a directory called `weizmann` in `datasets`, and download the zip files and segmentation masks from the [Weizmann dataset website](http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html).

## Models and Inference Methods

The `models` subdirectory contains three different inference methods that can be used with MDMM (or MDMM-like) architectures:

- `dmm.py` implements the MDMM with Bidirectional Factorized Variational Inference, a.k.a. Backward Forward Variational Inference (BFVI), as described in our paper. Refer to the included docstrings for a full list of options.

- `dks.py` implements the MDMM with the RNN-based structured inference networks described by [Krishnan et al](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14215). By providing different options to the constructor, one can use either forward or backward RNN networks, and toggle different methods for handling missing data. Refer to the docstrings for details.

- `vrnn.py` implements a multimodal version of the Variational Recurrent Neural Network (VRNN) described by [Chung et al](https://papers.nips.cc/paper/5653-a-recurrent-latent-variable-model-for-sequential-data). This is similar to using `dks.py` with a forward RNN.

## Training

The training code for the Spirals dataset can be run by calling:
```python spirals.py```
Default hyper-parameters are used, run `python spirals.py -h` for a full list of options.

The training code for the Weizmann dataset can be run by calling:
```python weizmann.py```
Again, default hyper-parameters are used, run `python weizmann.py -h` for a full list of options.

To specify which inference method to use, use the `--model` flag with either `dmm` or `dks`. To specify which modalities to load and train on, use the `--modalities` flag. To visualize predictions while training, add the `--visualize` flag. Pretrained models can be evaluated by adding `--load PATH/TO/MODEL`.

An abstract `Trainer` class can be found in `trainer.py`, allowing training code to easily written for other multimodal sequential datasets.

## Experiments

Ray Tune can be used to easily run experiments across multiple sets of hyper-parameters over multiple trials. Make sure `ray` is installed for this to work. Install `tensorboard` and `tensorflow` as well if you would like to visualize the loss curves via Tensorboard.

### Comparing different inference methods on a range of tasks

For the Spirals dataset:
```python -m experiments.spirals_suite --trial_cpus N --trial_gpus N```

For the Weizmann dataset:
```python -m experiments.weizmann_suite --trial_cpus N --trial_gpus N```

### Learning with uniformly random missing data

For the Spirals dataset:
```python -m experiments.spirals_partial --trial_cpus N --trial_gpus N```

For the Weizmann dataset:
```python -m experiments.weizmann_partial --trial_cpus N --trial_gpus N```

### Semi-supervised learning

Semi-supervised learning refers to learning where some sequences have entire modalities removed.

For the Spirals dataset:
```python -m experiments.spirals_semisup --trial_cpus N --trial_gpus N```

For the Weizmann dataset:
```python -m experiments.weizmann_semisup --trial_cpus N --trial_gpus N```

## Examples

Refer to the [paper](https://arxiv.org/abs/1905.13570) for example results.

## Bugs & Questions

Feel free to raise issues, or email xuan [at] mit [dot] edu with questions.
