# obj-particles-latentrep
Learning latent structure for particle-based representation of objects

Get data and pretrained pcvae model from https://drive.google.com/open?id=1bGZn9PPWca8Yh57ju2W8NyY9Khrtf2rd

## Bag dataset
The cloth movement data can be downloaded from https://drive.google.com/file/d/1ObYcbD2ujX6YCUhECuKEn3Y_bewfBOcn/view?usp=sharing

(updated version: https://drive.google.com/file/d/1EPWb3pdJg5J8GfaDwxu8mVuRkC86sTpV/view?usp=sharing)

The bag (1145 vertices) is manipulated with 8 different operations with random speed. 
> In the h5 dataset file, the 2nd dim corresponds to z axis and the 3rd dim corresponds to the y axis. The "pcvae/viscloth.ipynb" shows some examples in the dataset. The "pcvae/loadH5Data.py" file includes a simple example of training pcvae with the cloth data (without temporal information).

0. move left with random speed.
1. move right with random speed.
2. left then right
3. close the bag
4. open the bag
5. close then open
6. move with random direction in the x-y plane 
7. drop onto the table and record the deformation
