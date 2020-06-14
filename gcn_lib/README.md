
# GCN library

gcn_lib contains the implementation of multiple **graph convolutions** and useful modules in [PyTorch](https://pytorch.org/). 
For now, it is mainly used for point cloud processing. 
Later, gcn_lib will include the GCN operations for image processing as well. 
gcn_lib is maintained by [Guocheng Qian](https://www.gcqian.com/). 

## Introduction

This repository contains the implementation of multiple **graph convolutions** and useful modules in [PyTorch](https://pytorch.org/).

## Installation

This implementation has been tested on Ubuntu 18.04 with CUDA 10.0, PyTorch (0.4 to 1.4). 
One only needs to install PyTorch.  For example, 
```
conda install -y pytorch=1.4.0 torchvision cudatoolkit=10.0 python=3.7 -c pytorch
```

## How to use

We provide some examples:
 
* [Scene Segmentation](./doc/scene_segmentation_guide.md): Instructions to train KP-FCNN on a scene segmentation 
 task (S3DIS).

## Cite
Please consider cite our paper:
```
@article{li2019deepgcns,
  title={Deepgcns: Making gcns go as deep as cnns},
  author={Li, Guohao and M{\"u}ller, Matthias and Qian, Guocheng and Delgadillo, Itzel C and Abualshour, Abdulellah and Thabet, Ali and Ghanem, Bernard},
  journal={arXiv preprint arXiv:1910.06849},
  year={2019}
}
```


## License
Our code is released under MIT License (see LICENSE file for details).

## Updates
* 15/05/2020: Initial release.
