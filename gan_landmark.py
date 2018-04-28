import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot

MODE = 'wgan' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 100000 # How many generator iterations to train for 
#OUTPUT_DIM = 784# Number of pixels in MNIST (28*28)
OUTPUT_DIM = 3072# Number of pixels in MNIST (28*28)

lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    print(noise.shape)

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    print(output.shape)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    print(output.shape)
    output = tf.nn.relu(output)
    print(output.shape)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])
    print(output.shape)

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    print(output.shape)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    print(output.shape)
    output = tf.nn.relu(output)
    print(output.shape)

    output = output[:,:,:8,:8]
    print(output.shape)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    print(output.shape)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    print(output.shape)
    output = tf.nn.relu(output)
    print(output.shape)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)
    print(output.shape)
    output = tf.nn.sigmoid(output)
    print(output.shape)
    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs):
    output = tf.reshape(inputs,[-1,3 ,32 ,32 ])

    output = lib.ops.conv2d.Conv2D('Discriminator.1',3,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, 
        labels=tf.ones_like(disc_fake)
    ))

    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_fake, 
        labels=tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=disc_real, 
        labels=tf.ones_like(disc_real)
    ))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

# For saving samples
fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples = Generator(128, noise=fixed_noise)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    lib.save_images.save_images(
        samples.reshape((128, 32, 32,3)), 
        'samples_landmark{}.png'.format(frame)
    )

#import h5py    
#data_dir='./'
#filepath = data_dir + "normalizedData6k_7k.h5"
#trainset = {}
#f = h5py.File(filepath)
#for k, v in f.items():
#    trainset[k] = np.array(v)
#print(trainset['X'][0])
#def scale(x, feature_range=(-1, 1)):
#    # scale to (0, 1)
#    x = ((x - x.min())/(255 - x.min()))
#    
#    # scale to feature_range
#    min, max = feature_range
#    x = x * (max - min) + min
#    return x
#trainset['X']=scale(trainset['X'])
#print(trainset['X'][0])

trainset=np.load("normalizedData6k_7k.npy")
#print(trainset[:,:,:,0])
trainset= np.rollaxis(trainset, axis=3)
trainset= np.reshape(trainset, [trainset.shape[0],OUTPUT_DIM])
print(trainset.shape)
#features=trainset
#labels=trainset['y']
#print(features.shape)
#print(labels.shape)
##dx=tf.data.Dataset.from_tensor_slices(X).batch(BATCH_SIZE)
##iterator = dx.make_initializable_iterator()
#
##features_placeholder = tf.placeholder(features.dtype, features.shape)
##labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
##dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder)).batch(BATCH_SIZE)
#
#features_placeholder = tf.placeholder(features.dtype, features.shape)
#dataset = tf.data.Dataset.from_tensor_slices(features_placeholder).batch(BATCH_SIZE)
#iterator = dataset.make_initializable_iterator()

# Dataset iterator
#train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)

length = abs(trainset.shape[0]-(trainset.shape[0]%BATCH_SIZE))
print (length)
my_trainset=[]



def inf_train_gen():
    while True:
        for i in xrange(0,length,BATCH_SIZE):
            yield trainset[i:i+BATCH_SIZE,:]
	#for images,targets in train_gen():
        #for images in trainset['X']:
        #    yield images

# Train loop
with tf.Session() as session:

    session.run(tf.initialize_all_variables())
    #session.run(iterator.initializer, feed_dict={features_placeholder: features})
    gen = inf_train_gen()

    for iteration in xrange(ITERS):
        start_time = time.time()

        if iteration > 0:
            _ = session.run(gen_train_op)

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data = gen.next()
            #print(_data.shape)
            #_data2=iterator.get_next()
            #print(_data[0]) 
            #print(_data2.shape) 
            #print(_data2[0]) 
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            #dev_disc_costs = []
            #for images,_ in dev_gen():
            #    _dev_disc_cost = session.run(
            #        disc_cost, 
            #        feed_dict={real_data: images}
            #    )
            #    dev_disc_costs.append(_dev_disc_cost)
            #lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration, _data)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()
