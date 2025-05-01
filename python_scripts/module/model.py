import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import layers
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
        config = super().get_config()
        config.update({"padding": self.padding})
        return config


class AttentionBlock(layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers.
        groups: Number of groups to be used for Groupalization layer.
    """

    def __init__(self, units, groups=-1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.groups = groups
        seed = np.random.randint(0, 1000000)
        self.kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=seed)

        self.norm = GNorm(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=self.kernel_init)
        self.key = layers.Dense(units, kernel_initializer=self.kernel_init)
        self.value = layers.Dense(units, kernel_initializer=self.kernel_init)
        self.proj = layers.Dense(units, kernel_initializer=self.kernel_init)

    def call(self, inputs):
        batch_size, height, width, _ = tf.shape(inputs)
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc,bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj


class Generator:
    def __init__(
        self,
        filters=None,
        num_resid=6,
        num_inputs=2,
        skip=False,
        attention=False,
        spat_dim=128,
        name="RES",
    ):
        if filters is None:
            filters = [128, 256]
        self.filters = filters
        self.num_resid = num_resid
        self.num_inputs = num_inputs
        self.skip = skip
        self.attention = attention
        self.spat_dim = spat_dim
        self.name = name

        # Weight initializers
        seed = np.random.randint(0, 1000000)
        self.kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=seed)
        self.gamma_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=seed)

    def upsample(self, x, filters, activation, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False):
        result = keras.Sequential()
        result.add(
            layers.Conv2DTranspose(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                kernel_initializer=self.kernel_init,
                use_bias=use_bias,
            )
        )
        result.add(GNorm(groups=-1, gamma_initializer=self.gamma_initializer))
        if activation:
            result.add(activation)
        return result(x)

    def downsample(self, x, filters, activation, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False):
        result = keras.Sequential()
        result.add(
            layers.Conv2D(
                filters,
                kernel_size,
                strides=strides,
                kernel_initializer=self.kernel_init,
                padding=padding,
                use_bias=use_bias,
            )
        )
        result.add(GNorm(groups=-1, gamma_initializer=self.gamma_initializer))
        if activation:
            result.add(activation)
        return result(x)

    def residual_block(self, x, activation, kernel_size=(3, 3), strides=(1, 1), padding="valid", use_bias=False):
        dim = x.shape[-1]
        result = keras.Sequential()

        result.add(ReflectionPadding2D())
        result.add(
            layers.Conv2D(
                dim,
                kernel_size,
                strides=strides,
                kernel_initializer=self.kernel_init,
                padding=padding,
                use_bias=use_bias,
            )
        )
        result.add(GNorm(groups=-1, gamma_initializer=self.gamma_initializer))
        result.add(activation)

        result.add(ReflectionPadding2D())
        result.add(
            layers.Conv2D(
                dim,
                kernel_size,
                strides=strides,
                kernel_initializer=self.kernel_init,
                padding=padding,
                use_bias=use_bias,
            )
        )
        result.add(GNorm(groups=-1, gamma_initializer=self.gamma_initializer))
        return result(x)

    def get_generator(self, name="gen"):
        sampling = len(self.filters)

        input_img_size = [self.spat_dim, self.spat_dim, 1]
        img_input1 = layers.Input(shape=input_img_size, name=f"{name}_img_input1")
        img_input2 = layers.Input(shape=input_img_size, name=f"{name}_img_input2")

        x = layers.Concatenate(axis=-1)([img_input1, img_input2])

        x = ReflectionPadding2D(padding=(3, 3))(x)
        x = layers.Conv2D(self.filters[0] // 2, (7, 7), kernel_initializer=self.kernel_init, use_bias=False)(x)
        x = GNorm(groups=-1, gamma_initializer=self.gamma_initializer)(x)
        x = layers.Activation("relu")(x)
        x_first = x

        # Downsampling
        down_stack = []
        for i in range(sampling):
            x = self.downsample(x, filters=self.filters[i], activation=layers.Activation("relu"))
            if self.skip:
                down_stack.append(x)

        # Residual blocks
        for _ in range(self.num_resid):
            x_res = x
            x = self.residual_block(x, activation=layers.Activation("relu"))
            if self.attention:
                x = AttentionBlock(self.filters[-1], groups=-1)(x)
            x = layers.add([x_res, x])

        # Upsampling
        self.filters.reverse()
        down_stack.reverse()
        for j in range(sampling):
            if self.skip:
                x = layers.Concatenate()([x, down_stack[j]])
            x = self.upsample(x, self.filters[j] // 2, activation=layers.Activation("relu"))

        self.filters.reverse()
        if self.skip:
            x = layers.Concatenate()([x, x_first])

        # Final block
        x = ReflectionPadding2D(padding=(3, 3))(x)
        x = layers.Conv2D(1, (7, 7), padding="valid")(x)
        x = layers.Activation("tanh")(x)

        model = keras.models.Model([img_input1, img_input2], x, name=name)
        return model

class Img2Img(keras.Model):
    
    def __init__(self,generator_G,augment=False):
        super().__init__()
        self.gen_G = generator_G
        
        self.augment= augment
        self.data_augmentation = keras.Sequential([
              layers.RandomFlip("horizontal"),
              layers.RandomRotation((-0.2, 0.2),fill_mode='constant'),
              layers.GaussianNoise(0.015)
          ])


    def compile(
        self,
        gen_G_optimizer,
        gen_loss_fn,
        metric,
    ):
        super().compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.metric = metric

    @staticmethod
    def norm_0_1(arr):
        return (arr + 1.0) / 2.0

    def train_step(self, batch_data):

        if self.augment:
            x  = batch_data[:,0]
            y  = batch_data[:,1]
            z  = batch_data[:,2]

            conc = tf.concat([x,y,z], axis=-1)
            
            aug = self.data_augmentation(conc,training=True)

            input_1  = aug[:,:,:,0:1]
            input_2  = aug[:,:,:,1:2]
            target  = aug[:,:,:,2:]
        
        else:
            input_1  = batch_data[:,0]
            input_2  = batch_data[:,1]
            target  = batch_data[:,2]
        

        with tf.GradientTape(persistent=True) as tape:
            fake = self.gen_G([input_1,input_2], training=True)

            # Generator adversarial loss
            gen_G_loss = self.generator_loss_fn(target, fake)

        # Get the gradients for the generator
        grads_G = tape.gradient(gen_G_loss, self.gen_G.trainable_variables)

        # Update the weights of the generator
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        ssim = tf.image.ssim(
            self.norm_0_1(fake),
            self.norm_0_1(target),
            max_val=1.0,
        )

        self.metric.update_state(ssim)

        return {
            "G_loss": gen_G_loss,
            "SSIM": self.metric.result(),
        }


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, model, suffix, out_dir, ckpt):
        super().__init__()
        self.model = model
        self.suffix = suffix
        self.out_dir = out_dir
        self.ckpt = ckpt
        self.log_file_name = None

    def on_train_begin(self, logs=None):
        self.log_file_name = f"log_{self.suffix}.csv"

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        os.chdir(self.out_dir)
        epoch = int(epoch) + 1

        g_loss = round(logs.get("G_loss", 0.0), 4)
        ssim = round(logs.get("SSIM", 0.0), 4)

        self.model.metric.reset_state()

        if (epoch % self.ckpt) == 0:
            g_filename = f"ckpt_G_{epoch}_{self.suffix}.keras"
            self.model.gen_G.save(g_filename)

        row_log = [epoch, g_loss, ssim]
        with open(self.log_file_name, "a", newline="") as f:
            writer = csv.writer(f)
            if epoch == 1:
                header = ["Epoch", "G_loss", "SSIM"]
                writer.writerow(header)
            writer.writerow(row_log)
