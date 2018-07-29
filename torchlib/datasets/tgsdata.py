


import os
import numpy as np


from .weightmaps import getweightmap
from .imageutl import  TGSProvide
from pytvision.datasets import utility 
from pytvision.transforms.aumentation import  ObjectImageMaskMetadataAndWeightTransform, ObjectImageMetadataTransform

import warnings
warnings.filterwarnings("ignore")


train = 'train'
validation = 'train'
test  = 'train'



class TGSDataset(object):
    '''
    Management for TGS Salt dataset
    '''

    def __init__(self, 
        base_folder, 
        sub_folder,  
        train=True,
        folders_images='images',
        folders_masks='masks',
        files = 'train.csv',
        metadata = 'depths.csv',
        ext='png',
        transform=None,
        count=None, 
        num_channels=3,
        ):
        """Initialization       
        """            
           
        self.data = TGSProvide.create( 
                base_folder, 
                sub_folder, 
                train,
                folders_images, 
                folders_masks,
                files,
                metadata,
                )
        
        self.transform = transform  
        self.count = count if count is not None else len(self.data)   
        self.num_channels = num_channels

    def __len__(self):
        return self.count

    def __getitem__(self, idx):   

        idx = idx % len(self.data)
        if self.data.train:
            name, image, mask, depth = self.data[idx] 
            image_t  = utility.to_channels(image, ch=self.num_channels )
            weight_t = getweightmap(mask)
            weight_t = weight_t[:,:,np.newaxis]
            mask_t   = np.zeros( (mask.shape[0], mask.shape[1], 2) )   
            mask_t[:,:,0] = (mask <= 0)
            mask_t[:,:,1] = (mask > 0)
            obj = ObjectImageMaskMetadataAndWeightTransform( image_t, mask_t, weight_t, np.array([depth])  )
        else:
            name, image, depth = self.data[idx] 
            image_t  = utility.to_channels(image, ch=self.num_channels )
            obj = ObjectImageMetadataTransform( image_t, np.array( (idx, depth ) )  )

        if self.transform: 
            obj = self.transform( obj )

        return obj.to_dict()


    
