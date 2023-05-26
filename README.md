# DTM extraction from DSM using Generative Adversarial Networks
This repository contains the code written in the context of a master thesis at the Université Libre de Bruxelles about the use of GANs for the generation of Digital Terrain Models (DTM) using Digital Terrain Models (DSM) and processed aerial photographs.
## Contents
This repository contains the jupyter notebooks, as well as python files used in the project. 
Dockerfile is an image definition for Docker

## Setting up the Docker environment
- Clone this repository on your computer
- Build the docker image : 
  - In a command prompt, go to the directory cloned from github
  - Command : ```docker build -t dtm .```
    (dtm will be the name of the image)
- Run a new Docker instancewith the comamnd : 
```
docker run --rm -p 0.0.0.0:6006:6006 -p 8888:8888 -v [DATA_DIR]:/home/student/data --gpus=all --name dtm dtm 
```
## Download dataset and model checkpoints
### Datasets
Create a new directory called "np_dataset", this directory will contain the datasets of DSMs and orthophotos used to extract DTMs. It contains also the DTMs used as supervision.<br>
The structure of this directory used while making this project was that it itself contains 2 directories called "512" and "256" (in reference to the spatial dimensions of the rasters").<br>
In these directories, the datasets were contained in directories called "train", "test" and "val" (in reference to their use)<br>
The rasters used as input data for the neural networks are saved numpy ndarrays, named "[id].npy", where [id] is a number between 1 and the total number of items in the dataset. These input raster are of size (RES, RES, n_channels), where RES is either 256 or 512, and where 
- n_channels = 1 if the raster contains only a DSM
- n_channels = 4 if the raster contains a DSM and an orthophotograph (the 1st channel is a float representing the DSM height and the 3 other channels are the RGB values)
If these input raster have corresponding DTMs used as supervision, they have the same structure (numpy array with shape (RES,RES)), and are saved with the name "dtm_[id].npy", where [id] is the same number as the corresponding DSM raster

The validation dataset used in the project can be downloaded at the url : https://zenodo.org/record/7971162. It should be stored in the directory "np_dataset/512/val/"

The datasets used in the project were sampled from the DSM, DTM and orthoplan of Brussels, available at https://datastore.brussels/web/urbis-download
The way the tif images downloaded on the Urbis website were used to generate the numpy dataset is shown in the jupyter notebook Preprocessing.ipynb

### Models
A selection of the deep neural networks trained for DTM extraction was put online and can be downloaded. They should be stored in a directory called "training_checkpoints/[model_name]" where the model name is the name given below. 
- pix_grad --> https://zenodo.org/record/7973800
- rgb_grad_fm --> https://zenodo.org/record/7973886
- seg/seg_model --> https://zenodo.org/deposit/7973867 
- gan_sem_map --> https://zenodo.org/record/7971961

## Generating DTMs
The jupyter notebook "main.ipynb" shows how to use the different models to generate DTMs using DSM and/or orthophotographs.<br>
It uses the dataset in directory "np_dataset/512/val", but any other dataset with the same structure can be used.

## Training a new model
The notebooks containing the code used in the project to traing new models are train_GAN.ipynb, train_GAN(+sem_map)/ipynb (adapted for the training of the GAN using as input the DSM and the semantic map of the ground) and train_seg.ipynb<br>
The code in these notebooks uses dataset in the form of tensorflow datasets, saved in the directory "tf_dataset/[set_type]_dataset_[RES]/", where [set_type] is either "train", "test" or "val" and [RES] is either "256" or "512". <br>
A tensorlow dataset compatible with the code in these notebook can be generated from a numpy dataset with the same structure as the one from the section above, using the fucntion "make_tf_dataset" from the preprocessing_fcts.py file.

## Results visualisation and statistics
The notebooks "visualize_gen.ipynb" and "visualize_seg.ipynb" contain the code used to visualize the results from trained neural networks, as well as measure their performance with diverse statistics

## Python files
The following files contain functions used in the different jupyter notebooks : 
- preprocessing_fcts : operations on datasets
- model_fcts : definition of networks architecture, class as well as function to compute the RMSE and MAE of the generated DTMs
- layer_fcts : definition of additional layers used in the definition of the networks
- loss_fcts : definition of operations used in the computation of losses
