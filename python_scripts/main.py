import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Custom imports
from illus_data import Illustris as get_illus
from model import Generator, Img2Img, CustomCallback

# Paths
home_dir = os.getcwd()
module_dir = os.path.join(home_dir, 'module')
os.chdir(module_dir)

# Constants
out_dir = os.path.join(home_dir, 'models')
img_dir = os.path.join(out_dir, 'imgs')
data_dir = os.path.join(home_dir, 'data')

# Parameters
total_epochs = int(sys.argv[1])
bands = list(sys.argv[2:])
spatial_dim = 128
crop = 54
suffix = 'initial_commit'

# Get Illustris data
illus = get_illus(bands, spatial_dim, crop, data_dir)

# Load IDs
os.chdir(data_dir)
train_ids = np.load('train_ids.npy')

train_files = illus.get_files(train_ids)
train_ds = illus.get_train_data(train_files)


# Custom loss function
def generator_loss_fn(real, fake):
    ssim_loss = 1 - tf.image.ssim(real, fake, max_val=1.0)
    mae_loss = tf.keras.losses.MeanAbsoluteError()(real, fake)
    return ssim_loss + mae_loss


# Build generator model
generator_network = generator(
    filters=[128, 256],
    num_resid=6,
    num_inputs=2,
    skip=False,
    attention=False,
).get_generator()

model = img2img(generator_network)

model.compile(
    gen_G_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    metric=keras.metrics.Mean(name="mean", dtype=None),
)

# Save every 8 epochs
ckpt = 8
callback = CustomCallback(model, suffix, out_dir, ckpt)

# Train the model
model.fit(train_ds, epochs=total_epochs, callbacks=[callback])
