import pickle as pkl
import time
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
import os
import sys
import h5py    
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


trdata=sys.argv[1]#'data0_1k.h5'
tedata=sys.argv[2]#'data0_1k.h5'
print(trdata)
print(tedata)
data_dir='./'
filepath = data_dir + trdata
trainset = {}
f = h5py.File(filepath)
for k, v in f.items():
    trainset[k] = np.array(v)
filepath = data_dir + tedata
testset = {}
f = h5py.File(filepath)
for k, v in f.items():
    testset[k] = np.array(v)
print(trainset['X'].shape)
print(trainset['y'])
print(trainset['y'].shape)
print(testset['X'].shape)
print(testset['y'])
print(testset['y'].shape)

trainset['y'] = (trainset['y']).astype(int).transpose()
testset['y'] = (testset['y']).astype(int).transpose()

numberOfTrainingExamples=trainset['X'].shape[3]
print(numberOfTrainingExamples)
numberOfTestExamples=testset['X'].shape[3]
print(numberOfTestExamples)

adjustment_test=np.min(testset['y'])-1
adjustment_train=np.min(trainset['y'])-1

for i in range(0,len(testset['y'])):
    testset['y'][i][0]=testset['y'][i][0]-adjustment_test

for i in range(0,len(trainset['y'])):
    trainset['y'][i][0]=trainset['y'][i][0]-adjustment_train

print(testset['y'])
print(trainset['y'])

def scale(x, feature_range=(-1, 1)):
    x = ((x - x.min())/(255 - x.min()))
    
    min, max = feature_range
    x = x * (max - min) + min
    return x

class Dataset:
    def __init__(self, train, test, val_frac=0.1, shuffle=True, scale_func=None):
        split_idx = int(len(test['y'])*(1 - val_frac))
        self.test_x = test['X'][:,:,:,:]
        self.test_y = test['y'][:]
        self.train_x, self.train_y = train['X'], train['y']
        
        self.label_mask = np.zeros_like(self.train_y)
        self.label_mask[0:-1] = 1
        
        self.train_x = np.rollaxis(self.train_x, axis=3)
        self.test_x = np.rollaxis(self.test_x, axis=3) 
        
        if scale_func is None:
            self.scaler = scale
        else:
            self.scaler = scale_func
        self.train_x = self.scaler(self.train_x)
        self.test_x = self.scaler(self.test_x)
        self.shuffle = shuffle
        
    def batches(self, batch_size, which_set="train"):
        x_name = which_set + "_x"
        y_name = which_set + "_y"

        num_examples = len(getattr(dataset, y_name))
        if self.shuffle:
            idx = np.arange(num_examples)
            np.random.shuffle(idx)
            setattr(dataset, x_name, getattr(dataset, x_name)[idx])
            setattr(dataset, y_name, getattr(dataset, y_name)[idx])
            if which_set == "train":
                dataset.label_mask = dataset.label_mask[idx]
        
        dataset_x = getattr(dataset, x_name)
        dataset_y = getattr(dataset, y_name)
        for ii in range(0, num_examples, batch_size):
            x = dataset_x[ii:ii+batch_size]
            y = dataset_y[ii:ii+batch_size]
            
            if which_set == "train":
                yield x, y, self.label_mask[ii:ii+batch_size]
            else:
                yield x, y


def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='input_real')
    
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    
    y = tf.placeholder(tf.int32, (None), name='y')
    
    label_mask = tf.placeholder(tf.int32, (None), name='label_mask')
    
    return inputs_real, inputs_z, y, label_mask


def generator(z, output_dim, reuse=False, alpha=0.2, training=True, size_mult=128):
    with tf.variable_scope('generator', reuse=reuse):
        
        x1 = tf.layers.dense(z, 4 * 4 * size_mult * 4)
        print(x1.shape)
        x1 = tf.reshape(x1, (-1, 4, 4, size_mult * 4))
        x1 = tf.layers.batch_normalization(x1, training=training)
        x1 = tf.maximum(alpha * x1, x1)
        print(x1.shape)
        x2 = tf.layers.conv2d_transpose(x1, size_mult * 2, 5, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=training)
        x2 = tf.maximum(alpha * x2, x2)
        print(x2.shape)
        x3 = tf.layers.conv2d_transpose(x2, size_mult, 5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(x3, training=training)
        x3 = tf.maximum(alpha * x3, x3)
        print(x3.shape)

        
        logits = tf.layers.conv2d_transpose(x3, output_dim, 2, strides=2, padding='valid')
        print(logits.shape)
        out = tf.tanh(logits)
        
        return out


def discriminator(x, reuse=False, alpha=0.2, drop_rate=0., num_classes=1000, size_mult=64):
    with tf.variable_scope('discriminator', reuse=reuse):
        print("discriminator x",x.shape)
        x = tf.layers.dropout(x, rate=drop_rate/2.5)
        
        # Input layer is ?x32x32x3
        x1 = tf.layers.conv2d(x, size_mult, 3, strides=2, padding='same')
        relu1 = tf.maximum(alpha * x1, x1)
        relu1 = tf.layers.dropout(relu1, rate=drop_rate) # [?x16x16x?]

        x2 = tf.layers.conv2d(relu1, size_mult, 3, strides=2, padding='same')
        bn2 = tf.layers.batch_normalization(x2, training=True) # [?x8x8x?]
        relu2 = tf.maximum(alpha * bn2, bn2)
        
        x3 = tf.layers.conv2d(relu2, size_mult, 3, strides=2, padding='same') # [?x4x4x?]
        bn3 = tf.layers.batch_normalization(x3, training=True)
        relu3 = tf.maximum(alpha * bn3, bn3)
        relu3 = tf.layers.dropout(relu3, rate=drop_rate)
        
        x4 = tf.layers.conv2d(relu3, 2 * size_mult, 3, strides=1, padding='same') # [?x4x4x?]
        bn4 = tf.layers.batch_normalization(x4, training=True)
        relu4 = tf.maximum(alpha * bn4, bn4)
        
        x5 = tf.layers.conv2d(relu4, 2 * size_mult, 3, strides=1, padding='same') # [?x4x4x?]
        bn5 = tf.layers.batch_normalization(x5, training=True)
        relu5 = tf.maximum(alpha * bn5, bn5)

        x6 = tf.layers.conv2d(relu5, 2 * size_mult, 3, strides=2, padding='same') # [?x2x2x?]
        bn6 = tf.layers.batch_normalization(x6, training=True)
        relu6 = tf.maximum(alpha * bn6, bn6)
        relu6 = tf.layers.dropout(relu6, rate=drop_rate)
        
        x7 = tf.layers.conv2d(relu5, filters=(2 * size_mult), kernel_size=3, strides=1, padding='valid')
        relu7 = tf.maximum(alpha * x7, x7)
        print("relu",relu7.shape)
        features = tf.reduce_mean(relu7, axis=[1,2])
        print("features-shape",features.shape)
        class_logits = tf.layers.dense(features, num_classes)
        print("class_logits",class_logits)
        gan_logits = tf.reduce_logsumexp(class_logits, 1)
        print("gan_logits",gan_logits)
        out = tf.nn.softmax(class_logits) # class probabilities for the 10 real classes plus the fake class
        print("out",out.shape)
        return out, class_logits, gan_logits, features

def model_loss(input_real, input_z, output_dim, y, num_classes, label_mask, alpha=0.2, drop_rate=0., smooth=0.1):
    
    g_size_mult = 28
    d_size_mult = 64
    
    g_model = generator(input_z, output_dim, alpha=alpha, size_mult=g_size_mult)
    d_on_data = discriminator(input_real, alpha=alpha, drop_rate=drop_rate, size_mult=d_size_mult)
    
    d_model_real, class_logits_on_data, gan_logits_on_data, data_features = d_on_data
    
    d_on_samples = discriminator(g_model, reuse=True, alpha=alpha, drop_rate=drop_rate, size_mult=d_size_mult)
    d_model_fake, class_logits_on_samples, gan_logits_on_samples, sample_features = d_on_samples
    
    real_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_data, 
                                                            labels=tf.ones_like(gan_logits_on_data) * (1 - smooth)))
    
    fake_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits_on_samples, 
                                                         labels=tf.zeros_like(gan_logits_on_samples)))
    
    unsupervised_loss = real_data_loss + fake_data_loss
    
    y = tf.squeeze(y)
    suppervised_loss = tf.nn.softmax_cross_entropy_with_logits(logits=class_logits_on_data,
                                                                  labels=tf.one_hot(y, num_classes, dtype=tf.float32))
    
    label_mask = tf.squeeze(tf.to_float(label_mask))
    
    suppervised_loss = tf.reduce_sum(tf.multiply(suppervised_loss, label_mask))
    
    suppervised_loss = suppervised_loss / tf.maximum(1.0, tf.reduce_sum(label_mask))
    d_loss = unsupervised_loss + suppervised_loss
    
    data_moments = tf.reduce_mean(data_features, axis=0)
    sample_moments = tf.reduce_mean(sample_features, axis=0)
    g_loss = tf.reduce_mean(tf.abs(data_moments - sample_moments))

    pred_class = tf.cast(tf.argmax(class_logits_on_data, 1), tf.int32)
    eq = tf.equal(tf.squeeze(y), pred_class)
    correct = tf.reduce_sum(tf.to_float(eq))
    masked_correct = tf.reduce_sum(label_mask * tf.to_float(eq))
    
    return d_loss, g_loss, correct, masked_correct, g_model,class_logits_on_data,pred_class,tf.squeeze(y)

def model_opt(d_loss, g_loss, learning_rate, beta1):
    discriminator_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES , scope='discriminator')
    generator_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    
    d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1, name='d_optimizer').minimize(d_loss, var_list=discriminator_train_vars)
    g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1, name='g_optimizer').minimize(g_loss, var_list=generator_train_vars)

    shrink_lr = tf.assign(learning_rate, learning_rate * 0.9)
    
    return d_train_opt, g_train_opt, shrink_lr

class GAN:
    def __init__(self, real_size, z_size, learning_rate, num_classes=1000, alpha=0.2, beta1=0.5):
        tf.reset_default_graph()
        
        self.learning_rate = tf.Variable(learning_rate, trainable=False)
        inputs = model_inputs(real_size, z_size)
        self.input_real, self.input_z, self.y, self.label_mask = inputs
        self.drop_rate = tf.placeholder_with_default(.5, (), "drop_rate")
        
        loss_results = model_loss(self.input_real, self.input_z,
                                  real_size[2], self.y, num_classes,
                                  label_mask=self.label_mask,
                                  alpha=0.2,
                                  drop_rate=self.drop_rate)
        self.d_loss, self.g_loss, self.correct, self.masked_correct, self.samples, self.class_logits_on_data,self.pred_class,self.actual_class = loss_results
        
        self.d_opt, self.g_opt, self.shrink_lr = model_opt(self.d_loss, self.g_loss, self.learning_rate, beta1)

def train(net, dataset, epochs, batch_size, figsize=(5,5)):
    
    saver = tf.train.Saver()
    sample_z = np.random.normal(0, 1, size=(50, z_size))

    samples, train_accuracies, test_accuracies = [], [], []
    steps = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            print("Epoch",e)
            
            t1e = time.time()
            num_examples = 0
            num_correct = 0
            for x, y, label_mask in dataset.batches(batch_size):
                assert 'int' in str(y.dtype)
                steps += 1
                num_examples += label_mask.sum()

                batch_z = np.random.normal(0, 1, size=(batch_size, z_size))

                t1 = time.time()
                _, _, correct = sess.run([net.d_opt, net.g_opt, net.masked_correct],
                                         feed_dict={net.input_real: x, net.input_z: batch_z,
                                                    net.y : y, net.label_mask : label_mask})
                t2 = time.time()
                num_correct += correct

            sess.run([net.shrink_lr])
            
            
            train_accuracy = num_correct / float(num_examples)
            
            print("\t\tClassifier train accuracy: ", train_accuracy)
            
            num_examples = 0
            num_correct = 0
            for x, y in dataset.batches(batch_size, which_set="test"):
                assert 'int' in str(y.dtype)
                num_examples += x.shape[0]

                correct, = sess.run([net.correct], feed_dict={net.input_real: x,
                                                   net.y : y,
                                                   net.drop_rate: 0.})
                num_correct += correct
            
            test_accuracy = num_correct / float(num_examples)
            print("\t\tClassifier test accuracy", test_accuracy)
            print("\t\tStep time: ", t2 - t1)
            t2e = time.time()
            print("\t\tEpoch time: ", t2e - t1e)
            
            
            gen_samples = sess.run(
                                   net.samples,
                                   feed_dict={net.input_z: sample_z})
            print(gen_samples.shape)
            samples.append(gen_samples)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            
        loc='./checkpoints/generatortrain'+trdata+'test'+tedata+'.ckpt'
        saver.save(sess, loc)
    name='samplestrain'+trdata+'test'+tedata+'.pkl'
    with open(name, 'wb') as f:
        pkl.dump(samples, f)
    
    return train_accuracies, test_accuracies, samples

real_size = (32,32,3)
z_size = 100
learning_rate = 0.0003

net = GAN(real_size, z_size, learning_rate)

dataset = Dataset(trainset, testset)
batch_size = 128
epochs =50
train_accuracies, test_accuracies, samples = train(net,
                                                   dataset,
                                                   epochs,
                                                   batch_size,
                                                   figsize=(5,5))

saver = tf.train.Saver()
num_classes=1000
features = np.empty([numberOfTestExamples,num_classes])#, dtype=int)#object)
labels = np.empty([1,numberOfTestExamples],dtype=int)#, dtype=int)#object)
net_predclass = np.empty([numberOfTestExamples],dtype=int)
net_actualclass = np.empty([numberOfTestExamples],dtype=int)

numo=0
with tf.Session() as sess:
  loc='./checkpoints/generatortrain'+trdata+'test'+tedata+'.ckpt'
  saver.restore(sess, loc)
  print("Model restored.")
  num_examples = 0
  num_correct = 0
  index=0
  for x, y in dataset.batches(batch_size, which_set="test"):
    assert 'int' in str(y.dtype)
    num_examples += x.shape[0]
    correct,my_class_logits_on_data,net_pred,net_actual = sess.run([net.correct,net.class_logits_on_data,net.pred_class,net.actual_class], feed_dict={net.input_real: x,
                                                   net.y : y,
                                                   net.drop_rate: 0.})
    features[index:index+len(y)]=my_class_logits_on_data
    net_predclass[index:index+len(y)]=net_pred
    net_actualclass[index:index+len(y)]=net_actual
    num_correct += correct
    index=index+len(y)
    numo=numo+len(y)
    
test_accuracy = num_correct / float(num_examples)

mypred_class=np.argmax(features,1).astype(int)
my_correct=np.sum(mypred_class==net_actualclass)
print(num_correct)
print(my_correct)
print("\t\tClassifier test accuracy", test_accuracy)
print(my_class_logits_on_data.shape)
print(numo)
print(features)
print(features.shape)
print(net_predclass)
print(net_actualclass)
print(mypred_class)
name2='featureTrain'+trdata+'Test'+tedata+'.h5'
hf=h5py.File(name2,'w')
hf.create_dataset('X',data=features)
hf.create_dataset('y',data=net_actualclass)
hf.close

