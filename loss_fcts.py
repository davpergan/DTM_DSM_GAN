import tensorflow as tf



'''
Loss functions
'''
LAMBDA = 100


def generator_loss(disc_generated_output, gen_output, target):
    
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + LAMBDA * l1_loss

    return total_gen_loss, gan_loss, l1_loss

# loss function implementing feature matching
def generator_fm_loss(disc_generated_output, disc_real_output, gen_output, target):
    
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    gan_loss = loss_object(tf.ones_like(disc_generated_output[1]), disc_generated_output[1])
    
    fm_loss = tf.reduce_mean(tf.abs(disc_generated_output[0] - disc_real_output[0]))

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + fm_loss + LAMBDA * l1_loss

    return total_gen_loss, gan_loss, l1_loss, fm_loss


def generator_fm_loss2(disc_generated_output, disc_real_output, gen_output, target):
    
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    gan_loss = loss_object(tf.ones_like(disc_generated_output[-1]), disc_generated_output[-1])
    
    fm_loss = tf.reduce_mean(tf.abs(disc_generated_output[0] - disc_real_output[0])) + tf.reduce_mean(tf.abs(disc_generated_output[1] - disc_real_output[1])) + tf.reduce_mean(tf.abs(disc_generated_output[2] - disc_real_output[2]))

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + fm_loss + LAMBDA * l1_loss

    return total_gen_loss, gan_loss, l1_loss, fm_loss

# feature matching and a gradient component

def generator_fm_grad_loss(disc_generated_output, disc_real_output, gen_output, target):
    
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    gan_loss = loss_object(tf.ones_like(disc_generated_output[-1]), disc_generated_output[-1])
    
    fm_loss = tf.reduce_mean(tf.abs(disc_generated_output[0] - disc_real_output[0])) + tf.reduce_mean(tf.abs(disc_generated_output[1] - disc_real_output[1])) + tf.reduce_mean(tf.abs(disc_generated_output[2] - disc_real_output[2]))

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    im_loss = tf.reduce_mean(grad(gen_output, target))

    total_gen_loss = gan_loss + fm_loss + LAMBDA * (l1_loss + im_loss)

    return total_gen_loss, gan_loss, l1_loss, fm_loss, im_loss



def discriminator_loss(disc_real_output, disc_generated_output):
    
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss
    

def grad(arr1, arr2):
    """
    Computes the gradient and normal components of the loss function, as defined in the paper "Dtm extraction from dsm using a multi-scale dtm fusion strategy based on deep learning" by Amirkolaee et al
    """
    
    dx1 = tf.expand_dims(tf.expand_dims(tf.pad(arr1[0,:,:,0][1:,] - arr1[0,:,:,0][:-1,], [[0,1],[0,0]]), axis=0), axis=-1)
    dy1 = tf.expand_dims(tf.expand_dims(tf.pad(arr1[0,:,:,0][:,1:] - arr1[0,:,:,0][:,:-1], [[0,0],[0,1]]), axis=0), axis=-1)
    
    dx2 = tf.expand_dims(tf.expand_dims(tf.pad(arr2[0,:,:,0][1:,] - arr2[0,:,:,0][:-1,], [[0,1],[0,0]]), axis=0), axis=-1)
    dy2 = tf.expand_dims(tf.expand_dims(tf.pad(arr2[0,:,:,0][:,1:] - arr2[0,:,:,0][:,:-1], [[0,0],[0,1]]), axis=0), axis=-1)
    
    grad_im = tf.sqrt(tf.square(dy1 - dy2)+ tf.square(dx1 - dx2) + 1.0e-12)
    
    num_im = dx1*dx2 + dy1*dy2 + 1
    denum_im = tf.sqrt((tf.square(dx1) + tf.square(dy1) + 1) * (tf.square(dx2) + tf.square(dy2) + 1))

    norm_im = 1- num_im/denum_im

    
    
    return grad_im + norm_im


def generator_grad_loss(disc_generated_output, gen_output, target):
    
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    im_loss = tf.reduce_mean(grad(gen_output, target))

    total_gen_loss = gan_loss + (LAMBDA * (l1_loss + im_loss))

    return total_gen_loss, gan_loss, l1_loss, im_loss