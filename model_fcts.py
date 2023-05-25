import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf
import preprocessing_fcts as pp
from IPython import display
from keras.engine import data_adapter
from layer_fcts import *
from seg_models import *



'''
Networks models
'''



def Generator256(inp_channels, res, deconv=True):
    """
    Generator from original pix2pix network
    """
    if deconv:
        upsample = globals()['upsample1']
    else:
        upsample = globals()['upsample2']

    inputs = tf.keras.layers.Input(shape=[res, res, inp_channels])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)

    if deconv:
        last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 1)
    else:
        last = tf.keras.Sequential()

        last.add(tf.keras.layers.UpSampling2D(size=2, interpolation='nearest'))

        last.add(tf.keras.layers.Conv2D(1, 4, strides=1, padding='same',
                             kernel_initializer=initializer, activation='tanh'))

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return Generator(inputs=inputs, outputs=x)




def Generator512(inp_channels, res, deconv=True):
    """
    Added downsample and upsample blocs in attempt to improve GAN with 51x512x pixels input
    """
            
    inputs = tf.keras.layers.Input(shape=[res, res, inp_channels])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  
        downsample(128, 4), 
        downsample(256, 4), 
        downsample(512, 4),  
        downsample(512, 4), 
        downsample(512, 4),  
        downsample(512, 4),  
        downsample(512, 4), 
        downsample(512, 4),
    ]

    up_stack = [
        upsample1(512, 4, apply_dropout=True),
        upsample1(512, 4, apply_dropout=True),  
        upsample1(512, 4, apply_dropout=True), 
        upsample1(512, 4, apply_dropout=True),  
        upsample1(512, 4), 
        upsample1(256, 4),  
        upsample1(128, 4),  
        upsample1(64, 4),  
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') 

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return Generator(inputs=inputs, outputs=x)




def naderi_gen(inp_channels, res, deconv=True): 
    """
    network with DRIB modules
    """
    inputs = tf.keras.layers.Input(shape=[res,res,inp_channels])
    
    if deconv:
        upsample = globals()['upsample1']
    else:
        upsample = globals()['upsample2']

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4, strides=1),  
    ]

    up_stack = [
        upsample(256, 4, strides=1, apply_dropout=True),  # (batch_size, 32, 32, 512)
        upsample(128, 4, apply_dropout=True),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    for i in range(8):
        x = tf.keras.layers.Add()([drib(x), x])

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return Generator(inputs=inputs, outputs=x)



def ASPP_gen(inp_channels, res, deconv=True):
    inputs = tf.keras.layers.Input(shape=[res,res,inp_channels])
    
    down_stack = [
        downsample(64, 4, apply_batchnorm=False), 
        downsample(128, 4), 
        downsample(256, 4),
        downsample(512, 4)
    ]

    up_stack = [
        upsample1(512, 4, apply_dropout=True),  
        upsample1(256, 4, apply_dropout=True), 
        upsample1(128, 4),  
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') 

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    x = ASPP(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return Generator(inputs=inputs, outputs=x)



def Discriminator256(inp_channels, res):
    """
    Discriminator from original pix2pix network
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[res, res, inp_channels], name='input_image')
    tar = tf.keras.layers.Input(shape=[res, res, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channel

    down1 = downsample(64, 4, apply_batchnorm=False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

def Discriminator512(inp_channels, res):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[res, res, inp_channels], name='input_image')
    tar = tf.keras.layers.Input(shape=[res, res, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, apply_batchnorm=False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    down4 = downsample(512, 4)(down3)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down4)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


"""
Discriminator model returning information used when implementing feature matching
"""

def Discriminator_fm_256(inp_channels, res):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[res, res, inp_channels], name='input_image')
    tar = tf.keras.layers.Input(shape=[res, res, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, apply_batchnorm=False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs= [down2, last])


def Discriminator_fm2_256(inp_channels, res):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[res, res, inp_channels], name='input_image')
    tar = tf.keras.layers.Input(shape=[res, res, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) 

    down1 = downsample(64, 4, apply_batchnorm=False)(x) 
    down2 = downsample(128, 4)(down1) 
    down3 = downsample(256, 4)(down2) 

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) 
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) 

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)

    return tf.keras.Model(inputs=[inp, tar], outputs= [down1, down2, down3, last])



def Discriminator_fm_512(inp_channels, res):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[res, res, inp_channels], name='input_image')
    tar = tf.keras.layers.Input(shape=[res, res, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) 

    down1 = downsample(64, 4, apply_batchnorm=False)(x)  
    down2 = downsample(128, 4)(down1) 
    down3 = downsample(256, 4)(down2)  
    down4 = downsample(512, 4)(down3)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down4) 
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) 

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) 

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) 

    return tf.keras.Model(inputs=[inp, tar], outputs= [down3, last])



''' 
Other
'''

class Generator(tf.keras.Model): 
    """
    Subclass of keras model, but keeps the parameter training=true when using the function 'predict'
    """
    
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=True)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        return self(x, training=True)
        
    


def generate_images(model, test_input, tar, inp_channels=1):
  prediction = model(test_input, training=True)
  if inp_channels != 1:
        test_input = test_input[:,:,:,0]
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()


def generate_images_rgb(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(20, 20))

    dsm = test_input[0,:,:,0]

    rgb = test_input[0,:,:,1:]
    rgb = tf.cast((rgb+1)*127, tf.int32)

    display_list = [rgb, dsm , tar[0], prediction[0]]
    title = ['Aerial image', 'Input Image', 'Ground Truth', 'Predicted Image']
    
    plt.subplot(1,4,1)
    plt.title(title[0])
    plt.imshow(display_list[0])
    plt.axis('off')

    for i in range(3):
        plt.subplot(1, 4, i+2)
        plt.title(title[i+1])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i+1] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

    
@tf.autograph.experimental.do_not_convert
def test_RMSE(preds): 
    """
    Computes the RMSE of a set of predictions
    """
    tbs = [] #list containing the value of highest and lowest pixel in each DSM (for normalization)
    res = preds[0].shape[1] #256 or 512
    test_ds, _ = pp.load_dataset(res, True) #dataset with DSMs and DTMs corresponding to the set of predictions
    tbs_ds = test_ds.map(lambda x, y : (tf.math.reduce_max(x[:,:,0]), tf.math.reduce_min(x[:,:,0])))
    
    for i in tbs_ds.as_numpy_iterator(): #fill the tbs list
        tbs.append(i)
    
    dtms = test_ds.map(lambda x, y: y)
    ssum = 0
    n = res**2

    N = str(preds.shape[0])
    i = 0
    for j in dtms.as_numpy_iterator():
        pred = preds[i]
        pred = pp.denormalize(pred, tbs[i][0], tbs[i][1])
        dif = np.sum((pred[:,:,0] - j)**2)
        ssum += dif
        i += 1
        #print(str(i) + '/' + N, end='\r')
    #print(i)
    return np.sqrt(ssum/(i*n))

@tf.autograph.experimental.do_not_convert
def test_L1(preds): 
    
    tbs = [] #list containing the value of highest and lowest pixel in each DSM (for normalization)
    res = preds[0].shape[1] #256 or 512
    test_ds, _ = pp.load_dataset(res, True) #dataset with DSMs and DTMs corresponding to the set of predictions
    tbs_ds = test_ds.map(lambda x, y : (tf.math.reduce_max(x[:,:,0]), tf.math.reduce_min(x[:,:,0])))
    
    for i in tbs_ds.as_numpy_iterator(): #fill the tbs list
        tbs.append(i)
    
    dtms = test_ds.map(lambda x, y: y)
    ssum = 0
    n = res**2

    N = str(preds.shape[0])
    i = 0
    for j in dtms.as_numpy_iterator():
        pred = preds[i]
        pred = pp.denormalize(pred, tbs[i][0], tbs[i][1])
        dif = np.sum(tf.abs((pred[:,:,0] - j)))
        ssum += dif
        i += 1
        #print(str(i) + '/' + N, end='\r')
    #print(i)
    return ssum/(i*n)

@tf.autograph.experimental.do_not_convert
def test_RMSE_norm(preds, test_ds, rgb): 
    tbs = []
    if rgb: 
        tbs_ds = test_ds.map(lambda x, y : (tf.math.reduce_max(x[:,:,0]), tf.math.reduce_min(x[:,:,0])))
    else:
        tbs_ds = test_ds.map(lambda x, y : (tf.math.reduce_max(x), tf.math.reduce_min(x)))
    
    for i in tbs_ds.as_numpy_iterator():
        tbs.append(i)
    
    dtms = test_ds.map(lambda x, y: y)
    ssum = 0
    n = preds.shape[1] * preds.shape[2]
    inp_channels = [i[0][0] for i in test_ds.take(1)][0].shape[-1]

    N = str(preds.shape[0])
    i = 0
    for j in dtms.as_numpy_iterator():
        pred = preds[i]
        normalized_dtm = pp.normalize(j, tbs[i][0], tbs[i][1])
        dif = np.sum((pred[:,:,0] - normalized_dtm)**2)
        ssum += dif
        i += 1
        #print(str(i) + '/' + N, end='\r')
    #print(i)
    return np.sqrt(ssum/(i*n))

@tf.autograph.experimental.do_not_convert
def test_L1_norm(preds, test_ds, rgb): 
    tbs = []
    if rgb: 
        tbs_ds = test_ds.map(lambda x, y : (tf.math.reduce_max(x[:,:,0]), tf.math.reduce_min(x[:,:,0])))
    else:
        tbs_ds = test_ds.map(lambda x, y : (tf.math.reduce_max(x), tf.math.reduce_min(x)))
    
    for i in tbs_ds.as_numpy_iterator():
        tbs.append(i)
    
    dtms = test_ds.map(lambda x, y: y)
    ssum = 0
    n = preds.shape[1] * preds.shape[2]
    inp_channels = [i[0][0] for i in test_ds.take(1)][0].shape[-1]

    N = str(preds.shape[0])
    i = 0
    for j in dtms.as_numpy_iterator():
        pred = preds[i]
        normalized_dtm = pp.normalize(j, tbs[i][0], tbs[i][1])
        dif = np.sum(tf.abs(pred[:,:,0] - normalized_dtm))
        ssum += dif
        i += 1
        #print(str(i) + '/' + N, end='\r')
    #print(i)
    return ssum/(i*n)



def load_model(name, net_type, res, activ, relu):
    net_fct = globals()[net_type]
    inputs = tf.keras.layers.Input(shape=[res, res, 4])
    net = net_fct(inputs, relu=relu, activ=activ)
    net_seg = Generator(inputs=inputs, outputs=net)
    
    checkpoint = tf.train.Checkpoint(model=net_seg)
    checkpoint.restore('training_checkpoints/seg/' + name + '/ckpt-1')
    return net_seg
