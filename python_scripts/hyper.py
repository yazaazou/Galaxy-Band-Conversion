import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import RMSprop
from keras import layers

from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.layers import InputSpec
from keras.layers import Layer
from keras.layers import GroupNormalization as GNorm

import cv2
import sys
import time
import datetime
import os
import numpy as np
import csv
import h5py as h5

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import image
from matplotlib.cbook import get_sample_data
from PIL import Image

import pandas as pd
from random import randint
from numpy import random


batch_size   =  1
totalEpochs  =  int(sys.argv[1])
n            =  int(sys.argv[2])
sampling_num =  int(sys.argv[3])
resid_num    =  int(sys.argv[4])

band1 = int(12)
band2 = int(20)

bands= str(band1) + '_'  + str(band2)
homeDir = '/home/yazaazou/thesis/hyperparameter/{}_sample/{}_blocks'.format(sampling_num,resid_num) 

metaDir= '/home/yazaazou/illustris/data/metadata'
dataDir= '/home/yazaazou/illustris/data'
outDir = homeDir + '/models'

os.chdir(dataDir)
ids= np.load('stratified_ids.npy')

bandDirDict= {1:'/galax_FUV_1',3:'/3_sdss_U',5:'/5_sdss_R',7:'/7_sdss_Z',8:'/8_Irac1',
             10:'/irac3_10',11:'/irac4_11',12:'/johns_U_12',13:'/13_john_B',
             14:'/cous_R_14',19:'/2Mass_H_19',20:'/johns_K_20',22:'/22_ACS_F435',
             23:'/23_ACS_F606',25:'/25_ACS_F850',28:'/28_f160w',32:'/32_Nircam_F150w',
             35:'/35_Nircam_F356w'}

band1Dir= dataDir +bandDirDict[band1]
band2Dir= dataDir +bandDirDict[band2]

#####
clip_min = -1.0
clip_max =  1.0


crop_factor = 54
spat_dim=  128
channels= 1
img_size = np.array([spat_dim, spat_dim, channels])
input_img_size = img_size

def get_files(max_index,ids, band1,band2):
    base_str='band_{}_id_{}.hdf5'
    os.chdir(dataDir)
    pair_list=[]

    os.chdir(band1Dir)
    band1DirList= os.listdir()

    os.chdir(band2Dir)
    band2DirList= os.listdir()

    inds= np.random.choice(ids,max_index,replace=False)

    for id in inds:
        name1 = base_str.format(band1, id)
        name2 = base_str.format(band2, id)
        try:
            if (name1 in band1DirList) and (name2 in band2DirList):
                pair_list.append([name1,name2])
            else:
                raise NameError
        except:
            pass

    return pair_list

def rescale(img):
    
    img= np.array(img)
    img= np.expand_dims(img,axis=-1)

    img= img[crop_factor:-crop_factor,crop_factor:-crop_factor]
    img = tf.image.resize(img, [spat_dim,spat_dim])
    
    img= img - tf.math.reduce_min(img) + 1e-10
    img= tf.math.asinh(img)
    img = tf.math.sqrt(img)
    img = (img - tf.math.reduce_min(img)) / (tf.math.reduce_max(img) - tf.math.reduce_min(img))
    img = (img*2) -1
    
    img = tf.clip_by_value(img, clip_min, clip_max)
    
    return img

def resize(img):
    return img+0.0

def train_preprocessing(img):
    img = resize(img)
    return img


def get_data(loc_list,epochs):
    num = len(loc_list)
    nan_ids=[]
    img_dim= spat_dim*spat_dim*channels
    arr= np.zeros(num*2*2*img_dim,dtype= np.float32).reshape([num*2,2,spat_dim,spat_dim,channels])

    i=0

    for pair in loc_list:
        os.chdir(band1Dir)
        f1= h5.File(pair[0],'r')
        os.chdir(band2Dir)
        f2= h5.File(pair[1],'r')

        f1_cam0= rescale(f1['camera0'])
        f2_cam0= rescale(f2['camera0'])
        cam0_nan= np.sum(np.isnan(f1_cam0)) + np.sum(np.isnan(f2_cam0))

        f1_cam2=rescale( f1['camera2'])
        f2_cam2=rescale( f2['camera2'])
        cam2_nan= np.sum(np.isnan(f1_cam2)) + np.sum(np.isnan(f2_cam2))

        if cam0_nan == 0:
            arr[i,0] = f1_cam0
            arr[i,1] = f2_cam0
        if cam2_nan == 0:
            arr[i+1,0] = f1_cam2
            arr[i+1,1] = f2_cam2
        if cam0_nan != 0:
            nan_ids.append(pair[0])

        if cam2_nan !=0:
            nan_ids.append(pair[1])
        i = i + 2

    dataset = tf.convert_to_tensor(arr,dtype= np.float32)
    del arr
    dataset= tf.data.Dataset.from_tensor_slices(dataset)

    train_ds = (
        dataset.map(train_preprocessing)
        .batch(1, drop_remainder=True)
        .repeat(epochs)
        .prefetch(tf.data.AUTOTUNE)
        )
    return train_ds


# Weights initializer for the layers.
seed= np.random.randint(0,1000000)
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02,seed = seed)

# Gamma initializer for instance normalization.
gamma_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02,seed = seed)
gamma_init =  keras.initializers.RandomNormal(mean=0.0, stddev=0.02,seed = seed)


class ReflectionPadding2D(layers.Layer):

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super().__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'padding': self.padding,
        })
        return config    
            
def residual_block(
    x,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="valid",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x=GNorm(groups=-1,gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x=GNorm(groups=-1,gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    #here
    x=GNorm(groups=-1,gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    #here
    x=GNorm(groups=-1,gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def get_resnet_generator(
    filters=64,
    num_downsampling_blocks=sampling_num,
    num_residual_blocks=resid_num,
    num_upsample_blocks=sampling_num,
    gamma_initializer=gamma_init,
    name=None,
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(
        x
    )

    x=GNorm(groups=-1,gamma_initializer=gamma_initializer)(x)

    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(channels, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model

def get_discriminator(
    filters=64, kernel_initializer=kernel_init, num_downsampling=3, name=None
):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer,
    )(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model


# Get the generators
gen_G = get_resnet_generator(name="generator_G")
gen_F = get_resnet_generator(name="generator_F")

# Get the discriminators
disc_X = get_discriminator(name="discriminator_X")
disc_Y = get_discriminator(name="discriminator_Y")


class CycleGan(keras.Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super().__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super().compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        real_x  = batch_data[:,0]
        real_y  = batch_data[:,1]

        with tf.GradientTape(persistent=True) as tape:
            
            fake_y = self.gen_G(real_x, training=True)
            
            fake_x = self.gen_F(real_y, training=True)

            # Cycle  x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle  y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adversarial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss_cycle": cycle_loss_G,
            "F_loss_cycle": cycle_loss_F,
            "G_loss_gen":gen_G_loss,
            "F_loss_gen":gen_F_loss,
            "G_loss": total_loss_G, 
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }
    
class CustomCallback(keras.callbacks.Callback):
    
    def __init__(self, model):
        self.model = model
    
    def on_train_begin(self, logs={}):
        
        self.G_losses_cycle = []
        self.F_losses_cycle=[]
        
        self.G_losses_gen = []
        self.F_losses_gen=[]
        
        self.G_losses = []
        self.F_losses=[]
        
        self.D_X_losses = []
        self.D_Y_losses = []
        
        self.logFileName = 'log_samp_{}_resid_{}.csv'.format(sampling_num,resid_num)
        
    def on_epoch_end(self, epoch, logs=None):
        
        os.chdir(outDir)
        epoch = int(epoch) + 1

        G_loss_cycle = round(logs.get("G_loss_cycle"),5)
        self.G_losses_cycle.append(G_loss_cycle)
        
        F_loss_cycle = round(logs.get("F_loss_cycle"),5)
        self.F_losses_cycle.append(F_loss_cycle)
        
        G_loss_gen = round(logs.get("G_loss_gen"),5)
        self.G_losses_gen.append(G_loss_gen)
        
        F_loss_gen = round(logs.get("F_loss_gen"),5)
        self.F_losses_gen.append(F_loss_gen)
        
        G_loss = round(logs.get("G_loss"),5)
        self.G_losses.append(G_loss)
        
        F_loss = round(logs.get("F_loss"),5)
        self.F_losses.append(F_loss)
        
        D_X_loss = round(logs.get("D_X_loss"),5)
        self.D_X_losses.append(D_X_loss)
        
        D_Y_loss = round(logs.get("D_Y_loss"),5)
        self.D_Y_losses.append(D_Y_loss)
        

        if (epoch % 4) == 0 :
            GFilename = 'ckpt_G_{}_samp_{}_resid_{}.keras'.format(epoch,sampling_num,resid_num)
            FFilename ='ckpt_F_{}_samp_{}_resid_{}.keras'.format(epoch,sampling_num,resid_num)
            XFilename = 'ckpt_X_{}_samp_{}_resid_{}.keras'.format(epoch,sampling_num,resid_num)
            YFilename = 'ckpt_Y_{}_samp_{}_resid_{}.keras'.format(epoch,sampling_num,resid_num)

            self.model.gen_G.save(GFilename)
            self.model.gen_F.save(FFilename)
            self.model.disc_X.save(XFilename)
            self.model.disc_Y.save(YFilename)

        row = [epoch, G_loss_cycle , F_loss_cycle, G_loss_gen , F_loss_gen\
               , G_loss, F_loss , D_X_loss , D_Y_loss]
        
        with open(self.logFileName, 'a') as f:
            writer = csv.writer(f)
            if epoch == 1:
                header = ['Epoch','G Cycle','F Cycle','G gen',\
                          'F gen','G total','F total','X total','Y total']
                writer.writerow(header)
            writer = csv.writer(f)
            writer.writerow(row)
                       
# Loss function for evaluating adversarial loss
adv_loss_fn = keras.losses.MeanSquaredError()

# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss

# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5

# Create cycle gan model
cycle_gan_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
)

files= get_files(max_index=n,ids=ids, band1=band1,band2=band2)
steps= len(files*2)
dataset = get_data(files,totalEpochs)
cycle_gan_model.fit(dataset,epochs=totalEpochs,steps_per_epoch=steps,callbacks=[CustomCallback(cycle_gan_model)])

