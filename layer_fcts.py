"""
Functions used in "model_fcts.py" to define layers or blocs of layers in networks
"""
import tensorflow as tf



def downsample(filters, size, strides=2, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def downsample2(filters, size, strides=2, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.ReLU())

  return result


def upsample1(filters, size, strides=2, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


def upsample2(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()

    result.add(tf.keras.layers.UpSampling2D(2, interpolation='nearest'))

    result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result



def drib1(filters):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, 1, strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.ReLU())

    return result

def drib2(filters, dil):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, 3, strides=1, dilation_rate=(dil,dil), padding='same',
                             kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.ReLU())

    return result

def drib3(filters):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2D(filters, 1, strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())


    return result


def drib(x):
    
    layer1 = drib1(256)
    layer1b = drib1(256)
    layer1c = drib1(256)
    
    layer2 = drib2(256,1)
    layer2b = drib2(256,3)
    layer2c = drib2(256,5)
    
    layer3 = drib3(512)
    
    x1 = layer1(x)
    x1 = layer2(x1)
    
    x2 = layer1b(x)
    x2 = layer2b(x2)
    
    x3 = layer1c(x)
    x3 = layer2c(x3)
    
    x = tf.keras.layers.Concatenate()([x1,x2, x3])
   
    x = layer3(x)
    return x


def convolution_block(
    block_input,
    num_filters=512,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False):

    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(block_input)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)


def ASPP(dspp_input): # Code from : https://keras.io/examples/vision/deeplabv3_plus/
    dims = dspp_input.shape
    x = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = tf.keras.layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, num_filters=256, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=18)

    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, num_filters=256, kernel_size=1)
    return output