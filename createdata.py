
import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt 

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn


from pytvision.transforms.aumentation import  ObjectImageMetadataTransform
from pytvision.transforms import transforms as mtrans

from torchlib.datasets import tgsdata
from torchlib.datasets.tgsdata import TGSDataset
from torchlib.datasets.imageutl import TGSProvide, TGSExProvide

def maggradient(image):
    gI = np.gradient(image)
    divI =  (gI[0]**2 + gI[1]**2)**0.5 
    mg  = np.sum(divI>0.001)/np.prod(divI.shape)
    return mg


def main():
    pathnamedataset  = '~/.kaggle/competitions/tgs-salt-identification-challenge'
    data  = TGSProvide.create(  pathnamedataset, sub_folder='train', train=True, files='train.csv' )
    print( len(data) )

    # train metadata generate
    metadata = []
    for i in tqdm(range( len(data) )):
        name, image, mask, depth = data[ i ]        
        mg     = maggradient(image)
        area   = mask.sum()
        mean   = image.mean()
        std    = image.std()           
        metadata.append( { 
            'name':name, 
            'depth':depth, 
            'mg':mg, 
            'area':area, 
            'mean':mean,
            'std':std,
            'bool': mg > 0.2 ,
            }  )                    
    metadata = pd.DataFrame( metadata )
    metadata.head()

    filename=os.path.join(pathnamedataset,'metadata_train.csv')
    metadata.to_csv(filename, index=None, encoding='utf-8')
    print('save train metadata ...')

    # test metadata generate

    data  = TGSProvide.create(  pathnamedataset, sub_folder='test', train=False, files='sample_submission.csv' )
    print( len(data) )

    metadata = []
    for i in tqdm(range( len(data) )):
        name, image, depth = data[ i ]        
        mg     = maggradient(image)
        mean   = image.mean()
        std    = image.std()           
        metadata.append( { 
            'name':name, 
            'depth':depth, 
            'mg':mg, 
            'mean':mean,
            'std':std,
            'bool':True,
            }  )        
            
    metadata = pd.DataFrame( metadata )
    metadata.head()

    filename=os.path.join(pathnamedataset,'metadata_test.csv')
    metadata.to_csv(filename, index=None, encoding='utf-8')
    print('save test metadata ...')

    print('DONE!!!')




if __name__ == '__main__':
    main()