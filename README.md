# Kalman Variational Auto-Encoder
> A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning.

This repository contains the code for Kalman Variational Auto-Encoders introduced in  _A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning_.

<div style="text-align:center"><img src="assets/kvae_figure.png" width="300"></div>

## Installation instructions
The Kalman Variational Auto-Encoder (KVAE) package can be installed running
```
# Install requirements
pip install tensorflow-gpu numpy pandas matplotlib seaborn

# Clone the kvae repository and install it
git clone https://github.com/simonkamronn/kvae
cd kvae
pip install -e .
```

### Dependencies
- Python >= 2.7 or 3.5
- Tensorflow >= 1.1

## Data generation
The bouncing ball data is generated running `kvae/utils/box.py`, which will save video sequences to the `data` folder in your root directory.
This code depends on pygame and the pymunk physics engine
```
pip install pygame pymunk
cd kvae/utils/
python box.py
```

## Usage Example
In the `examples` folder execute the command
```
python boxed_ball.py  --gpu=0
```
to train a model using the default dataset and parameters. To train a model with different parameters run
```
python boxed_ball.py  --help
```
to view the configuration options or look in `kvae/utils/config.py`.

## License
* Free software: MIT license
