import tensorflow as tf
from layer_fcts import *



def PixUnet(inputs, relu=True, activ='sigmoid'):
        
    if relu:
        downsample = globals()['downsample2']
    else:
        downsample = globals()['downsample']

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
        upsample1(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample1(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample1(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample1(512, 4),  # (batch_size, 16, 16, 1024)
        upsample1(256, 4),  # (batch_size, 32, 32, 512)
        upsample1(128, 4),  # (batch_size, 64, 64, 256)
        upsample1(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)

    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                     strides=2,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     activation=activ)  

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

    return x


def CustomUnet512(inputs):

    down_stack = [
        downsample2(64, 4, apply_batchnorm=False),
        downsample2(128, 4),  
        downsample2(256, 4),  
        downsample2(512, 4),  
        downsample2(512, 4),  
        downsample2(512, 4),  
        downsample2(512, 4),  
        downsample2(512, 4),  
        downsample2(512, 4),
    ]

    up_stack = [
        upsample1(512, 4, apply_dropout=True),
        upsample1(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample1(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample1(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample1(512, 4),  # (batch_size, 16, 16, 1024)
        upsample1(256, 4),  # (batch_size, 32, 32, 512)
        upsample1(128, 4),  # (batch_size, 64, 64, 256)
        upsample1(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)

    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                     strides=2,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     activation='sigmoid')  # (batch_size, 256, 256, 3)

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

    return x




def Custom_naderi_net(inputs):
    
    down_stack = [
        downsample2(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample2(128, 4),  # (batch_size, 64, 64, 128)
        downsample2(256, 4),  # (batch_size, 32, 32, 256)
        downsample2(512, 4, strides=1)
    ]

    up_stack = [
        upsample1(256, 4, strides=1, apply_dropout=True),  # (batch_size, 32, 32, 512)
        upsample1(128, 4, apply_dropout=True),  # (batch_size, 64, 64, 256)
        upsample1(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='sigmoid')  # (batch_size, 256, 256, 3)

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

    return x

        
def Naderi_net512(inputs):
    
    down_stack = [
        downsample2(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample2(128, 4),  # (batch_size, 64, 64, 128)
        downsample2(256, 4),  # (batch_size, 32, 32, 256)
        downsample2(512, 4,),
        downsample2(512, 4, strides=1)
    ]

    up_stack = [
        upsample1(256, 4, strides=1, apply_dropout=True), 
        upsample1(256, 4, apply_dropout=True),
        upsample1(128, 4),  # (batch_size, 64, 64, 256)
        upsample1(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='sigmoid')  # (batch_size, 256, 256, 3)

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

    return x
    
    
    
def Custom_naderi_net2(inputs):
    
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512,4),
        downsample(512, 4, strides=1)
    ]

    up_stack = [
        upsample1(512,4, strides=1, apply_dropout=True),
        upsample1(256, 4, apply_dropout=True),  # (batch_size, 32, 32, 512)
        upsample1(128, 4, apply_dropout=True),  # (batch_size, 64, 64, 256)
        upsample1(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='sigmoid')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    for i in range(5):
        x = tf.keras.layers.Add()([drib(x), x])

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return x



def ASPP_net(inputs, relu=True, activ='sigmoid'):
        
    if relu:
        downsample = globals()['downsample2']
    else:
        downsample = globals()['downsample']
        
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),
        downsample(1024, 4)
        
    ]

    up_stack = [
        upsample1(512, 4, apply_dropout=True),
        upsample1(256, 4, apply_dropout=True) ,  # (batch_size, 32, 32, 512)
        upsample1(128, 4),  # (batch_size, 64, 64, 256)
        upsample1(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation=activ)  # (batch_size, 256, 256, 3)

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

    return x


def ASPP_net512(inputs):
    
    down_stack = [
        downsample2(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample2(128, 4),  # (batch_size, 64, 64, 128)
        downsample2(256, 4),  # (batch_size, 32, 32, 256)
        downsample2(512, 4),
        downsample2(512, 4, strides=1)
    ]

    up_stack = [
        upsample1(256, 4, strides=1, apply_dropout=True),
        upsample1(256, 4, apply_dropout=True),  # (batch_size, 32, 32, 512)
        upsample1(128, 4),  # (batch_size, 64, 64, 256)
        upsample1(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='sigmoid')  # (batch_size, 256, 256, 3)

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

    return x



def ASPP_net2(inputs):
    
    down_stack = [
        downsample2(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample2(128, 4),  # (batch_size, 64, 64, 128)
        downsample2(256, 4),  # (batch_size, 32, 32, 256)
        downsample2(512, 4, strides=1)
    ]

    up_stack = [
        upsample1(256, 4, strides=1, apply_dropout=True),  # (batch_size, 32, 32, 512)
        upsample1(128, 4, apply_dropout=True),  # (batch_size, 64, 64, 256)
        upsample1(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='sigmoid')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    x = ASPP(x)
    x = ASPP(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return x
