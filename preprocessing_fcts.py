import matplotlib.pyplot as plt
import pathlib
import time
import datetime
import os
import numpy as np
import random
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1177989684
import tensorflow as tf
from IPython import display


def load_np_image(image_id, res, dataset, rgb): 
    """
    load an array (dsm or dsm+rgb) from a saved np dataset as well as the supervision (dtm) 
    """
    
    input_image = np.load('np_dataset/' + res + '/'+ dataset + '/' + image_id + '.npy')
    if not rgb:
        input_image = np.expand_dims(input_image[:,:,0],-1)
    real_image = np.load('np_dataset/' + res + '/'+ dataset + '/dtm_' + image_id + '.npy')
    return input_image, real_image


def load_ds(RES, ds_type,rgb):
    ds_type = ds_type.decode()
    path = 'np_dataset/' + str(RES) + '/' + ds_type
    id_list = [name[4:-4] for name in os.listdir(path) if 'dtm' in name]
    N = str(len(id_list))
    i = 0
    for num in id_list:
        print(i, end='\r')
        i += 1
        input_image, real_image = load_np_image(num, str(RES), ds_type, rgb=rgb)
        yield (input_image, real_image)

        
def make_tf_dataset(RES, ds_type, rgb):
    """
    Makes a tensorflow dataset from a numpy dataset
    Inputs : 
    - RES : either 256 or 512
    - ds_type : 'train', 'test' or 'val'
    - rgb : if True -> DSM+orthophotos, if False -> only DSM
    The np dataset should be in the directory 'np_dataset/[RES]/[ds_type]/'
    
    """
    if rgb:
        n_channels = 4
    else:
        n_channels = 1
    train_dataset_rgb = tf.data.Dataset.from_generator(load_ds, args=(RES,ds_type,rgb), output_signature=(
    tf.TensorSpec(shape=(RES,RES, n_channels), dtype=tf.float32),
    tf.TensorSpec(shape=(RES,RES), dtype=tf.float32,)))
    return train_dataset_rgb


def normalize(array, top, bottom):
    return ((array - bottom) / (top - bottom) - 0.5) * 2


def denormalize(array, top, bottom):
    return (array / 2.0 + 0.5) * (top - bottom) + bottom
        
    
def normalize_rgb(image): 
    """
    normalization for the rgb part
    """
    image = (image / 127.5) - 1
    return image 


def make_seg_ds(res, set_type, threshold = 0.01):
    """
    Function to make a dataset used in the training of a segmentation network
    It requires to have saved a tensorflow dataset in a directory 'tf_dataset/[set_type]_dataset_[res]'
    This tensorflow dataset can be produced from the npy dataset with the function 'make_tf_dataset'
    Inputs : 
     - res : 256 or 512
     - set_type : 'train' (or 'test' or 'val')
     - threshold : float used to produce the ground truth segmentation. the threshold is applied on the ndsm
    Output : 
    a tensorflow dataset, with 2 compontents per item : 
     - an array with dims (res, res, 4), where the 4 channels are the DSM + RGB channels (normalized between -1.0 and 1.0)
     - a semantic map, with dims (res,res)
    """
    ds_path = 'tf_dataset/' + set_type + '_dataset_' + str(res)
    ds = tf.data.Dataset.load(ds_path)
    ds = ds.map(lambda x, y: (x, x[:,:,0] - y))
    ds = ds.map(lambda x,y: (x, tf.cast(tf.math.greater(y, tf.constant(threshold)), dtype='float32')))
    ds = ds.map(lambda x, y : (x, y, (tf.math.reduce_max(x[:,:,0]), tf.math.reduce_min(x[:,:,0]))))
    ds = ds.map(lambda x, y, z:(tf.concat([tf.expand_dims(normalize(x[:,:,0], z[0], z[1]), axis=-1),normalize_rgb(x[:,:,1:])], axis=-1), tf.expand_dims(y, axis=-1)))
    
    return ds


def load_tf_ds(res):
    """
    Function to load saved training and testing tensorflow dataset (produced for example from npy dataset with the function 'make_tf_ds'
    Returns a tuple of datasets (train and test)
    """
    
    dir_path = 'tf_dataset/'
    
    train_path, test_path = dir_path + 'train_dataset_', dir_path + 'test_dataset_'
    
    #train_path, test_path = train_path + 'rgb_', test_path + 'rgb_'
    
    train_dataset = tf.data.Dataset.load(train_path + str(res))
    test_dataset = tf.data.Dataset.load(test_path + str(res))
    
    return train_dataset, test_dataset


@tf.autograph.experimental.do_not_convert
def normalize_tf_dataset(ds, rgb=False):
    """
    Function that takes as input a tensorflow dataset (output of 'make_tf_ds') and returns a dataset with the same shape but with values normalized between -1.0 and 1.0
    Input
     - ds : tf dataset
     - rgb : if True, 4 channels included (DSM+RGB), if False just DSM
    """
    ds = ds.map(lambda x, y : (x, y, (tf.math.reduce_max(x[:,:,0]), tf.math.reduce_min(x[:,:,0]))))
    if rgb:
        ds = ds.map(lambda x, y, z:(tf.concat([tf.expand_dims(normalize(x[:,:,0], z[0], z[1]), axis=-1),normalize_rgb(x[:,:,1:])], axis=-1), tf.expand_dims(normalize(y, z[0], z[1]), axis=-1)))
    else:
        ds = ds.map(lambda x, y, z:(tf.expand_dims(normalize(x[:,:,0], z[0], z[1]), axis=-1), tf.expand_dims(normalize(y, z[0], z[1]), axis=-1)))
    return ds



def load_dataset(res, rgb):
    #load the dataset corresponding to the resolution given in the input (and with or without the orthophotographs)
    raw_train_dataset, raw_test_dataset = load_tf_ds(res)
    test_dataset = normalize_tf_dataset(raw_test_dataset, rgb=rgb)
    test_dataset = test_dataset.batch(1)
    return raw_test_dataset, test_dataset




        








