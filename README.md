![Poster](https://user-images.githubusercontent.com/55981657/116153120-20a0d700-a6e7-11eb-98f5-cdfdb80f91b5.png)

### Prerequisites
install the following packages:

```
sudo pip install git+https://github.com/daavoo/pyntcloud
sudo pip3 install PyWavefront
sudo conda install -c conda-forge tensorflow 
sudo pip install tensorboardX
```

### Siamese Net
to use the siamese net launch the jupyter notebook:

execute.ipynb

this notebook contains the preprocessing pipieline and also the procedure for training, validation, testing and retrieval.

#### Remark
SiameseNet.py contains the final architecture with autoencoder structure. Our baseline model is in SiameseNet_Baseline.py