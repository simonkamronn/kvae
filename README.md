# Kalman Variational Auto-Encoder
> A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning.

We introduce the Kalman variational auto-encoder, a framework for unsupervised learning of sequential data that disentangles two latent representations: an object’s representation, coming from a recognition model, and a latent state describing its dynamics. The recognition model is represented by a convolutional variational auto-encoder and the latent dynamics model as a linear Gaussian state space model (LGSSM).
<div style="text-align:center"><img src="assets/kvae_figure.png" width="300"></div>

## Main Results
 We motivated the KVAE with an example of a bouncing ball, and use it here to demonstrate the model’s ability to separately learn a recognition and dynamics model from video, and use it to impute missing data. We simulate 5000 sequences of 20 time steps each of a ball moving in a two-dimensional box, where
each video frame is a 32x32 binary image. A video sequence is visualised as a single image in figure 4d, with the ball’s darkening color reflecting the incremental frame index. We compare the generation and imputation performance of the KVAE with two recurrent neural network (RNN) models that are based on the same auto-encoding (AE) architecture as the KVAE and and show that KVAE is better at simulating future frames (figure 3a and 3b).
<div style="text-align:center"><img src="assets/results.png" width="800"></div>

## Videos
Videos of simulations are available for many of the experiments here: [link](https://drive.google.com/drive/folders/0B7BmG5ubHI3UeDNLbVVXWDRVUnM?usp=sharing).

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
The bouncing ball data is generated running `kvae/datasets/box.py`, which will save video sequences to the `data` folder in your root directory.
This code depends on pygame and the pymunk physics engine.
```
pip install pygame pymunk
cd kvae/datasets/
python box.py
```
In the same folder you can find code to generate the data for the other environments (gravity, polygon and pong) presented in section 5.3 in the paper. Generated data of the Bouncing Ball experiment is available for download here: [link](https://drive.google.com/drive/folders/0B7BmG5ubHI3UeDNLbVVXWDRVUnM?usp=sharing).

## Usage Example
In the `examples` folder execute the command
```
python run_kvae.py  --gpu=0
```
to train a model using the default bouncing ball dataset and parameters. To train a model with different parameters run
```
python run_kvae.py  --help
```
to view the configuration options or look in `kvae/utils/config.py`.

## Citing
```
@article{,
    Author = {,Ulrich Paquet, Ole Winther},
    Title = {A Disentangled Recognition and Nonlinear Dynamics Model for Unsupervised Learning},
    Journal = {},
    Year = {2017}
}
```
## License
* Free software: MIT license
