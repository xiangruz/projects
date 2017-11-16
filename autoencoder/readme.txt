Autoencoder3D is a deep learning based framework for unsupervised clustering of Cellular Electron Cryo Tomography data

Please refer to our paper for more details:

A convolutional autoencoder approach for mining features in cellular electron cryo-tomograms and weakly supervised coarse segmentation
Xiangrui Zeng, Miguel Ricardo Leung, Tzviya Zeev-Ben-Mordehai, Min Xu
https://arxiv.org/abs/1706.04970


Key prerequisite:
EMAN2
keras
numpy
scipy



Installation: git clone ...



Input:

There are four inputs
1. A data file of CECT image patches, this data file should be prepared as follows:

d is an image patches data file.
d is a dictionary consists 'v_siz' and 'vs'.
d['v_siz'] is an numpy.ndarray specifying the shape of the image patch. For example, d['v_siz'] = array([32,32,32]).
d['vs'] is a dictionary with keys of uuids specifying each image patch.
d['vs'][an example uuid] is a dictionary consists 'center', 'id', and 'v'.
d['vs'][an example uuid]['center'] is the center of the image patch in the tomogram. For example, d['vs'][an example uuid]['center'] = [110,407,200].
d['vs'][an example uuid]['id'] is the specific uuid.
d['vs'][an example uuid]['v'] are voxel values of the image patch, which is an numpy.ndarray of shape d['v_siz']. 

2. A tomogram file in .rec format.

3. Whether the optional pose normalization step should be applied. Input should be True or False.

4. The number of clusters



Output:
1. A 'model' directory for the trained autoencoder models

2. A 'clus-center' directory for the resulting clusters. There should be two pickle files in 'clus-center'. 'kmeans.pickle' stores the uuids for each cluster. 'ccents.pickle' stores the decoded cluster centers. The 'fig' folder under 'clus-center' directory contains the 2D slices of decoded cluster center. User can use the figures as guide for manual selection.



Example usage: python autoencoder.py example/images.pickle example/tomogram.rec True 100




