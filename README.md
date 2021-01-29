# Graph Convolutional Layers and model of DeepSIBA: Implementation in Pytorch

# DeepSIBA: chemical structure-based inference of biological alterations using deep learning
### Christos Fotis<sup>1(+)</sup>, Nikolaos Meimetis<sup>1+</sup>, Antonios Sardis<sup>1</sup>, Leonidas G.Alexopoulos<sup>1,2(*)</sup>
 #### 1. BioSys Lab, National Technical University of Athens, Athens, Greece.
#### 2. ProtATonce Ltd, Athens, Greece.

(+)Equal contributions

(*)Correspondence to: leo@mail.ntua.gr

Original Github repository of the study:
> DeepSIBA: chemical structure-based inference of biological alterations using deep learning <br>
> Link: https://github.com/BioSysLab/deepSIBA <br>
> C.Fotis<sup>1(+)</sup>, N.Meimetis<sup>1+</sup>, A.Sardis<sup>1</sup>, LG. Alexopoulos<sup>1,2(*)</sup>

# DeepSIBA Approach
![figure1_fl_02](https://user-images.githubusercontent.com/48244638/80740035-212c7f00-8b20-11ea-9d97-300758595403.png)

## Clone
```bash
# clone the source code on your directory
$ git clone https://github.com/NickMeim/deepSIBA_pytorch.git
```
# Learning directory overview

This directory contains the gcnn layers and the model architecture to implement DeepSIBA in Pytorch. 

The NGF layers folders contain the source code to implement the graph convolution layers and the appropriate featurization. The code was adapted from https://github.com/keiserlab/keras-neural-graph-fingerprint.

The utility folder contains the following functions:

- Dataset loaders and generators
- A function to evaluate the performance of deepSIBA
- Custom layer and loss function to implement the Gaussian regression layer.
