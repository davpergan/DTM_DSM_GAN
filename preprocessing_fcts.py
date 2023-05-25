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


def load_np_image(image_id, res, dataset='train', rgb = False): 
    """
    load an array (dsm or dsm+rgb) from a saved np dataset as well as the supervision (dtm) 
    """
    
    if rgb:
        input_image = np.load('np_dataset/' + res + '/'+ dataset + '/rgb_' + image_id + '.npy')
    else:
        input_image = np.load('np_dataset/' + res + '/'+ dataset + '/' + image_id + '.npy')
    real_image = np.load('np_dataset/' + res + '/'+ dataset + '/dtm_' + image_id + '.npy')
    return input_image, real_image


def normalize_rgb(image): 
    """
    normalization for the rgb part
    """
    image = (image / 127.5) - 1
    return image 


def normalize(array, top, bottom):
    return ((array - bottom) / (top - bottom) - 0.5) * 2


def denormalize(array, top, bottom):
    return (array / 2.0 + 0.5) * (top - bottom) + bottom


def make_dataset(res, dataset='train', rgb = False, ): 
    """
    For a saved np dataset, loads all arrays ('input_image') and corresponding dtms ('real_image'), normalizes them and returns them in a list each
    the normalization here is a transormation that maps each dsm patch in a way to have its values comprised between -1 and 1
    the same tranformation is applied to the matching dtm
    """
    input_images = []
    real_images = []
    path = 'np_dataset/' + res + '/' + dataset
    id_list = [name[4:-4] for name in os.listdir(path) if 'dtm' in name]
    N = str(len(id_list))
    i = 0
    for num in id_list:
        input_image, real_image = load_np_image(num, res, dataset, rgb)
        
        if rgb: 
            input_image[:,:,1:] = normalize_rgb(input_image[:,:,1:])
            top, bottom = np.max(input_image[:,:,0]), np.min(input_image[:,:,0])
            real_image, input_image[:,:,0] = normalize(real_image, top, bottom), normalize(input_image[:,:,0], top, bottom)
        else:
            top, bottom = np.max(input_image), np.min(input_image)
            real_image, input_image = normalize(real_image, top, bottom), normalize(input_image, top, bottom)
            
        input_images.append(input_image)
        real_images.append(real_image)
        i += 1
        print(str(i) + '/' + N, end='\r')
        time.sleep(.01)
    return np.array(input_images), np.array(real_images)


def reshape_tf(item, res, rgb):
    # used to have the same dimensionality between the datasets with and without the rgb components
    dsm, dtm = item[0], item[1]
    res = int(res)
    if not rgb: 
        dsm = tf.reshape(dsm, (res,res,1))
    return (dsm, tf.reshape(dtm, (res,res,1)))


def load_rgb_ds(RES, ds_type):
    ds_type = ds_type.decode()
    path = 'np_dataset/' + str(RES) + '/' + ds_type
    id_list = [name[4:-4] for name in os.listdir(path) if 'dtm' in name]
    N = str(len(id_list))
    i = 0
    for num in id_list:
        print(i, end='\r')
        i += 1
        input_image, real_image = load_np_image(num, str(RES), ds_type, rgb=True)
        yield (input_image, real_image)

        
def make_rgb_tf_dataset(RES, ds_type):
    """
    Makes a tensorflow dataset from a numpy dataset
    Inputs : 
    - RES : either 256 or 512
    - ds_type : 'train', 'test' or 'val'
    The np dataset should be in the directory 'np_dataset/[RES]/[ds_type]/'
    
    """
    train_dataset_rgb = tf.data.Dataset.from_generator(load_rgb_ds, args=(RES,ds_type), output_signature=(
    tf.TensorSpec(shape=(RES,RES,4), dtype=tf.float32),
    tf.TensorSpec(shape=(RES,RES), dtype=tf.float32,)))
    return train_dataset_rgb
        
        
        
def load_tf_ds(res):
    
    dir_path = 'tf_dataset/'
    
    train_path, test_path = dir_path + 'train_dataset_', dir_path + 'test_dataset_'
    
    #train_path, test_path = train_path + 'rgb_', test_path + 'rgb_'
    
    train_dataset = tf.data.Dataset.load(train_path + str(res))
    test_dataset = tf.data.Dataset.load(test_path + str(res))
    
    return train_dataset, test_dataset


@tf.autograph.experimental.do_not_convert
def normalize_tf_dataset(ds, rgb=False):
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


def make_seg_ds(res, set_type, threshold = 0.01):
    ds_path = 'tf_dataset/' + set_type + '_dataset_' + str(res)
    ds = tf.data.Dataset.load(ds_path)
    ds = ds.map(lambda x, y: (x, x[:,:,0] - y))
    ds = ds.map(lambda x,y: (x, tf.cast(tf.math.greater(y, tf.constant(threshold)), dtype='float32')))
    ds = ds.map(lambda x, y : (x, y, (tf.math.reduce_max(x[:,:,0]), tf.math.reduce_min(x[:,:,0]))))
    ds = ds.map(lambda x, y, z:(tf.concat([tf.expand_dims(normalize(x[:,:,0], z[0], z[1]), axis=-1),normalize_rgb(x[:,:,1:])], axis=-1), tf.expand_dims(y, axis=-1)))
    
    return ds