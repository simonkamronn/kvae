# Kalman Variational Auto-Encoder
> A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning.

This repository contains the code for Kalman Variational Auto-Encoders introduced in  _A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning_.

<div style="text-align:center"><img src="assets/kvae_figure.png" width="300"></div>

## Installation instructions
To use the code it is assumed the Kalman Variational Auto-Encoder (KVAE) package is installed
```
# Install as package
python setup.py install

# Install in development mode
python setup.py develop
```

### Dependencies
- Python >= 2.7 or 3.5
- Tensorflow >= 1.1

## Usage Example
Generate data by modifying and running `kvae/utils/box.py` which will save video sequences to your the `data` folder in your root directory. In the `examples` folder execute the command
```
python boxed_ball.py  --dataset box_rnd
```
to train a model using the default dataset and parameters. To train a model with different parameters run
```
python boxed_ball.py  --help
```
to view the configuration options or look in `kvae/utils/config.py`.

## License
* Free software: MIT license
