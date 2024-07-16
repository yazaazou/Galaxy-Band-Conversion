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


homeDir= str(os.getcwd())

moduleDir=  homeDir + '/module'
os.chdir(moduleDir)
from illus_data import illustris_data as get_illus
from model import generator, img2img, CustomCallback


outDir = homeDir + '/models'
imgDir= outDir + '/imgs'

# specify data directory
dataDir= homeDir + '/data'

totalEpochs  =  int(sys.argv[1])
bands = list(sys.argv[2:])
spatial_dim = 128
crop = 54

suffix = 'initial_commit'
illus= get_illus(bands,spatial_dim,crop,dataDir)

# load IDS
os.chdir(dataDir)
train_ids= np.load('train_ids.npy')

train_files= illus.get_files(train_ids)
train_ds= illus.get_train_data(train_files)

# Build Model

# custom loss function
def generator_loss_fn(real,fake):
    ssim = tf.image.ssim
    ssim_loss= 1 - ssim(real,fake,max_val=1.0)

    mae =  tf.keras.losses.MeanAbsoluteError()
    mae_loss= mae(real,fake)
    return ssim_loss + mae_loss

generator_network = generator(filters=[128,256],num_resid=6,
                              num_inputs=2,skip=False,attention=False
                              ).get_generator()

model= img2img(generator_network)

model.compile(
    gen_G_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    metric=keras.metrics.Mean(name="mean", dtype=None)
    )

#save every 8 epochs
ckpt=8
call_back=  CustomCallback(model,suffix,outDir,ckpt)

model.fit(train_ds,epochs=totalEpochs, callbacks=[call_back])


