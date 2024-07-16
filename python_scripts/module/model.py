import sys
import os
import numpy as np
import csv
import h5py as h5

import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras import layers

from keras.layers import InputSpec
from keras.layers import GroupNormalization as GNorm



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

class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=-1, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)
        seed= np.random.randint(0,1000000)
        self.kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02,seed = seed)

        self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=self.kernel_init)
        self.key = layers.Dense(units, kernel_initializer=self.kernel_init)
        self.value = layers.Dense(units, kernel_initializer=self.kernel_init)
        self.proj = layers.Dense(units, kernel_initializer=self.kernel_init)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj

class generator:
    def __init__(self,
                 filters=[128,256],
                 num_resid=6,
                 num_inputs=2,
                 skip=False,
                 attention=False):
        
        
        self.filters=filters
        self.num_resid= num_resid
        self.num_inputs=num_inputs
        self.skip=skip
        self.attention= attention
        self.spat_dim=128
        self.name='RES'

        # Weights initializer for the layers.
        self.seed= np.random.randint(0,1000000)
        self.kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02,seed = self.seed)

        # Gamma initializer for instance normalization.
        self.gamma_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02,seed = self.seed)
        self.gamma_init =  keras.initializers.RandomNormal(mean=0.0, stddev=0.02,seed = self.seed)


    def upsample(self,
                 x,
                 filters,
                 g,
                 activation,
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 padding="same",
                 use_bias=False
                 ):
          
        result= tf.keras.Sequential()

        result.add(layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=self.kernel_init,
            use_bias=use_bias))
        
        result.add(GNorm(groups=-1,gamma_initializer=self.gamma_init))
        if activation:
            result.add(activation)

        return result(x)
    
    def downsample(self,
                   x,
                   filters,
                   g,
                   activation,
                   kernel_size=(3, 3),
                   strides=(2, 2),
                   padding="same",
                   use_bias=False
                   ):
        result= tf.keras.Sequential()
        result.add(layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            kernel_initializer=self.kernel_init,
            padding=padding,
            use_bias=use_bias))
        result.add(GNorm(groups=g ,gamma_initializer=self.gamma_init))
        if activation:
            result.add(activation)
        return result(x)
    

    def residual_block(self,
                       x,
                       activation,
                       g,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       padding="valid",
                       use_bias=False,
                       ):
        
        dim = x.shape[-1]

        result= tf.keras.Sequential()

        result.add(ReflectionPadding2D())
        result.add(layers.Conv2D(
            dim,
            kernel_size,
            strides=strides,
            kernel_initializer=self.kernel_init,
            padding=padding,
            use_bias=use_bias))
        result.add(GNorm(groups=g,gamma_initializer=self.gamma_init))
        result.add(activation)

        result.add(ReflectionPadding2D())
        result.add(layers.Conv2D(
            dim,kernel_size,
            strides=strides,
            kernel_initializer=self.kernel_init,
            padding=padding,
            use_bias=use_bias))
        result.add(GNorm(groups=g, gamma_initializer=self.gamma_init))
        return result(x)
        
    def get_generator(self,name='gen'):
        
        input_img_size=np.array([self.spat_dim, self.spat_dim, 1])

        img_input1 = layers.Input(shape=input_img_size, name=name + "_img_input1")
        img_input2 = layers.Input(shape=input_img_size, name=name + "_img_input2")
        img_input3 = layers.Input(shape=input_img_size, name=name + "_img_input3")
        img_input4 = layers.Input(shape=input_img_size, name=name + "_img_input4")
        img_input5 = layers.Input(shape=input_img_size, name=name + "_img_input5")
        img_input6 = layers.Input(shape=input_img_size, name=name + "_img_input6")

        inputs= [img_input1,img_input2,img_input3,img_input4,img_input5,img_input6]
        inputs= inputs[:self.num_inputs]
        
        if len(inputs)>1:
            x= layers.Concatenate(axis=-1)(inputs)
        elif len(inputs)==1:
            x= inputs[0]

        down_stack=[]
        sampling= len(self.filters)

        x = ReflectionPadding2D(padding=(3, 3))(x)
        x = layers.Conv2D(self.filters[0]/2, (7, 7), kernel_initializer=self.kernel_init, use_bias=False)(
            x
        )
        x=GNorm(groups=-1,gamma_initializer=self.gamma_initializer)(x)

        x = layers.Activation("relu")(x)
        x_first=x

        # Downsampling
        for i in range(sampling):
            x = self.downsample(x, filters=self.filters[i],g= -1, activation=layers.Activation("relu"))
            if self.skip:
                down_stack.append(x)

        # Residual blocks
        for _ in range(self.num_resid):
            x_res=x
            x = self.residual_block(x,activation=layers.Activation("relu"),g=-1)
            if self.attention:
                x = AttentionBlock(self.filters[-1], groups=-1)(x)
            x =  layers.add([x_res, x])
            
        # Upsampling
        self.filters.reverse()
        down_stack.reverse()
        for j in range(sampling):
            if self.skip:
                x=tf.keras.layers.Concatenate()([x, down_stack[j]])
            x = self.upsample(x, self.filters[j]/2, g=-1, activation=layers.Activation("relu"))

        self.filters.reverse()
        if self.skip:
            x=tf.keras.layers.Concatenate()([x, x_first])

        # Final block
        x = ReflectionPadding2D(padding=(3, 3))(x)

        x = layers.Conv2D(1, (7, 7), padding="valid")(x)
        x = layers.Activation("tanh")(x)

        model = keras.models.Model(inputs, x, name=name)
        return model

class img2img(keras.Model):
    def __init__(self,generator_G):
        super().__init__()
        self.gen_G = generator_G
    
    def compile(
        self,
        gen_G_optimizer,
        gen_loss_fn,
        metric ):

        super().compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.metric= metric

    def norm_0_1(self,arr):
        return (arr+1.0)/2.0

    def train_step(self, batch_data):

        inputs= [batch_data[:,:-1]]
        target  = batch_data[:,-1]
        
        with tf.GradientTape(persistent=True) as tape:
            
            fake = self.gen_G(inputs, training=True)
            
            # Generator adversarial loss
            gen_G_loss = self.generator_loss_fn(target,fake)
            
        # Get the gradients for the generators
        grads_G = tape.gradient(gen_G_loss, self.gen_G.trainable_variables)
        
        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        ssim = tf.image.ssim(self.norm_0_1(fake),self.norm_0_1(target), max_val=1.0)
        
        self.metric.update_state(ssim)

        return {
            "G_loss":gen_G_loss,
            "SSIM": self.metric.result()
            }    

class CustomCallback(keras.callbacks.Callback):
    
    def __init__(self, model,suffix,outDir,ckpt):
        super().__init__()
        self.model = model
        self.suffix= suffix
        self.outDir= outDir
        self.ckpt= ckpt
    
    def on_train_begin(self):
        self.logFileName = 'log_{}.csv'.format(self.suffix)
        
    def on_epoch_end(self, epoch, logs=None):
        
        os.chdir(self.outDir)
        epoch = int(epoch) + 1

        G_loss = round(logs.get("G_loss"),4)
        ssim= logs.get("SSIM")
        ssim = round(ssim,4)

        self.model.metric.reset_state()

        if (epoch % self.ckpt) == 0 :
            GFilename = 'ckpt_G_{}_{}.keras'.format(epoch,self.suffix)        
            self.model.gen_G.save(GFilename)      
        
        row_log = [epoch, G_loss,ssim]
        with open(self.logFileName, 'a') as f:
            writer = csv.writer(f)
            if epoch == 1:
                header = ['Epoch','G loss','ssim']
                writer.writerow(header)
            writer = csv.writer(f)
            writer.writerow(row_log)