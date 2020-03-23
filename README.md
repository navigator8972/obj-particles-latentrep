# obj-particles-latentrep
Learning latent structure for particle-based representation of objects

Get data and pretrained pcvae model from https://drive.google.com/open?id=1bGZn9PPWca8Yh57ju2W8NyY9Khrtf2rd

## Bag dataset
The cloth movement data can be downloaded from https://drive.google.com/file/d/13kDhpNBd8xOJw8Y4QF8dgSGkDkBj82Dl/view?usp=sharing

The bag is simulated with 1145 particles for original version, and the dense version contains 3435 points by resampling on the mesh. loadH5Data.py provides "PointCloudDataSetFromH5_3435_seq_interpolation" function to read the h5 dataset. When "preaction" = 0/4/5 (assign with a specific action id), the bags are set with a single initial state (without open/close angle). When "preaction" = None, the dataset is mixed with different initial state. During the movement, the bag is moved with the same speed and direction (the speed is added a slightly random noise).

> In the h5 dataset file, the 2nd dim corresponds to z axis and the 3rd dim corresponds to the y axis. The "pcvae/viscloth.ipynb" shows some examples in the dataset. The "pcvae/loadH5Data.py" file includes a simple example of training pcvae with the cloth data (without temporal information). At this stage, we only consider the "0" movement during the trajectory.

action id list:
0. move left with random speed.
1. move right with random speed.
2. left then right
3. close the bag
4. open the bag
5. close then open
6. move with random direction in the x-y plane 
7. drop onto the table and record the deformation
