
import os
import numpy as np
import PIL.Image
import scipy.misc

import cv2
import random
import csv
import pandas as pd
import operator

from pytvision.datasets.imageutl import dataProvide

from . import utility as utl



class TGSProvide(dataProvide):
    '''
    Mnagement for TGS dataset
    '''

    @classmethod
    def create(
        cls, 
        base_folder,
        sub_folder, 
        train=True,
        folders_images='images',
        folders_masks='masks',
        files = 'train.csv',
        metadata = 'depths.csv',
        ):
        '''
        Factory function that create an instance of TGSProvide and load the data form disk.
        '''
        provide = cls(base_folder, sub_folder, train, folders_images, folders_masks, files, metadata )
        provide.load_folders();
        return provide

    
    def __init__(self,
        base_folder,    
        sub_folder,     
        train=True,
        folders_images = 'images',
        folders_masks  = 'masks',
        files = 'train.csv',
        metadata = 'depths.csv',
        pref_image='',
        pref_label='',
        ):
        super(TGSProvide, self).__init__( );
        
        self.base_folder     = os.path.expanduser( base_folder )
        self.sub_folders     = sub_folder
        self.folders_images  = folders_images
        self.folders_masks   = folders_masks
        self.files           = files
        self.metadata        = metadata
        self.data            = []
        self.train           = train
    
    def getimagename(self, i):
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');
        self.index = i;
        return self.data[i][0]
        

    def __getitem__(self, i):
                
        #check index
        if i<0 and i>len(self.data): raise ValueError('Index outside range');
        self.index = i;

        #load data
        name           = self.data[i][0];  
        image_pathname = self.data[i][1]; 
        mask_pathname  = self.data[i][2];
        depth          = self.data[i][3];

        image = np.array(self._loadimage(image_pathname), dtype=np.uint8)

        if self.train:
            mask  =  PIL.Image.open(mask_pathname)  
            mask  = mask.convert('L').point(lambda x: 0 if x < 128 else 255, '1')
            mask  = np.array(mask)     
            return name, image, mask, depth
        else:
            return name, image, depth
            

    def load_folders(self):
        '''
        load file patch for disk
        '''
        
        self.data     = []                
        folder_path   = os.path.join(self.base_folder, self.sub_folders )
        folder_images = os.path.join(folder_path, self.folders_images)
        folder_masks  = os.path.join(folder_path, self.folders_masks)

        folder_files = os.path.join(self.base_folder, self.files )
        folder_meta  = os.path.join(self.base_folder, self.metadata )
        meta         = pd.read_csv( folder_meta )
        files        = pd.read_csv( folder_files )

        file_list = list(files['id'].values)
        depths = { i:z  for i,z in zip( meta['id'], meta['z'] ) }
   
        data =  [(f,
                os.path.join(folder_images,'{}.png'.format(f)),
                os.path.join(folder_masks,'{}.png'.format(f)),
                depths[f],
                ) for f in  file_list  ]    

        self.data = data

