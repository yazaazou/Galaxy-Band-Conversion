import tensorflow as tf
from tensorflow import keras
import numpy as np
import h5py as h5

import sys
import os

class illustris:
    def __init__(self,bands,spatial_dim,crop,dataDir):
        self.bands = bands
        self.band_num= len(bands)
        self.spatial_dim = spatial_dim
        self.crop = crop
        self.dataDir= dataDir

        self.clip_min=0.0
        self.clip_max=1.0
        self.img_size= np.array([spatial_dim, spatial_dim, 1])
        
        bandDirDict= {1:'/galax_FUV_1',2:'/Galax_NUV_2',3:'/3_sdss_U',5:'/5_sdss_R',7:'/7_sdss_Z',8:'/8_Irac1',
             10:'/irac3_10',11:'/irac4_11',12:'/johns_U_12',13:'/13_john_B',
             14:'/cous_R_14',19:'/2Mass_H_19',20:'/johns_K_20',21:'/2Mass_k_21',
             22:'/22_ACS_F435',23:'/23_ACS_F606',25:'/25_ACS_F850',28:'/28_f160w',
             32:'/32_Nircam_F150w', 35:'/35_Nircam_F356w'}
        bandDirs= []
        
        for b in self.bands:
            bandDirs.append(str(dataDir +bandDirDict[b]))

        self.bandDirs = bandDirs
        self.bandDirDict= bandDirDict

    def get_files(self,ids):
        base_str='band_{}_id_{}.hdf5'
        
        pair_list=[]
        DirList=[]

        for dir in self.bandDirs:
            os.chdir(dir)
            l= os.listdir()
            DirList.extend(l)

        inds= ids

        for id in inds: 
            names=[]
            include = True

            for b in self.bands:
                name = base_str.format(b, id)
                names.append(name)
                if name not in DirList:
                    include = False

            if include == False:
                continue  

            pair_list.append(names)
            
        return pair_list

    def rescale(self, img):
        
        img= np.array(img)
        img= np.expand_dims(img,axis=-1)

        img= img[self.crop:-self.crop,self.crop:-self.crop]
        img = tf.image.resize(img, [self.spat_dim,self.spat_dim])    
        img= img - tf.math.reduce_min(img) + 1e-10

        Q=3.5
        a=0.06
        img= tf.math.asinh(a*Q*img)/Q

        img = (img - tf.math.reduce_min(img)) / (tf.math.reduce_max(img) - tf.math.reduce_min(img))    
        img = tf.clip_by_value(img, self.clip_min, self.clip_max) 
        return img


    def get_train_data(self,loc_list):
        
        num = len(loc_list)
        img_dim= self.spatial_dim**2
        
        arr= np.zeros(num*4*self.band_num*img_dim,dtype= np.float32).reshape([num*2,self.band_num,self.spat_dim,self.spat_dim,1])
        
        for i,loc in zip( range(0, num, 4) , loc_list ):
            for j in range(self.band_num):
                img0,img1,img2,img3,nans= self.get_imgs(j,loc)
                if nans == 0:
                    arr[i,j]   = img0
                    arr[i+1,j] = img1
                    arr[i+2,j] = img2
                    arr[i+3,j] = img3

        dataset = tf.convert_to_tensor(arr,dtype= np.float32)
        del arr
        dataset= tf.data.Dataset.from_tensor_slices(dataset)

        train_ds = (
            dataset.batch(1, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
            )
        return train_ds


    def get_imgs(self,band_ind,loc):
        os.chdir(self.bandDirs[band_ind])
        f= h5.File(loc[band_ind],'r')

        # each fits file has 4 camera angles
        f_cam0= self.rescale(f['camera0'])
        f_cam1= self.rescale(f['camera1']) 
        f_cam2= self.rescale( f['camera2'])
        f_cam3= self.rescale(f['camera3'])
        
        #check for nans in rescaled images
        nans = np.sum(np.isnan(f_cam0)) +np.sum(np.isnan(f_cam1))+\
            np.sum(np.isnan(f_cam2))+np.sum(np.isnan(f_cam3))
        
        return f_cam0,f_cam1,f_cam2,f_cam3, nans

    def get_test_data(self,loc_list):
        num = len(loc_list)
        img_dim= self.spatial_dim**2
        
        arr= np.zeros(num*4*self.band_num*img_dim,dtype= np.float32).reshape([num*4,self.band_num,self.spat_dim,self.spat_dim,1])

        for i,loc in zip( range(0,num,4) , loc_list ):
            for j in range(self.band_num):
                img0,img1,img2,img3,nans= self.get_imgs(j,loc)
                if nans == 0:
                    arr[i,j] = img0
                    arr[i+1,j] = img1
                    arr[i+2,j] = img2
                    arr[i+3,j] = img3
            
        val_inp = arr[:,:int(self.band_num-1)]
        val_label= arr[:,-1:]
        del arr

        
        return val_inp, val_label
