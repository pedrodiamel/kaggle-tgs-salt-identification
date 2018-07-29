

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt 

import pytest


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn


from pytvision.transforms.aumentation import  ObjectImageMetadataTransform
from pytvision.transforms import transforms as mtrans

sys.path.append('../')
from torchlib.datasets import tgsdata
from torchlib.datasets.tgsdata import TGSDataset
from torchlib.segneuralnet import SegmentationNeuralNet
from torchlib.transforms import functional as F
from torchlib.utility import rle_encode, sigmoid



def test_model():

    
    project='../netruns'
    name='exp_tgs_unet_mcedice_adam_tgs_salt_001'
    path  = '~/.datasets/tgs_salt'
    pathmodel = os.path.join( project, name, 'models/model_best.pth.tar' )
    batch_size = 2
    workers=1
    cuda=False
    parallel=False
    gpu=0
    seed=1
    imsize=101

    dataset  = TGSDataset(  
        path, 
        'test', 
        num_channels=3,
        train=False, 
        files='sample_submission.csv',
        transform=transforms.Compose([
            mtrans.ToResizeUNetFoV(imsize, cv2.BORDER_REFLECT_101),
            mtrans.ToTensor(),
            mtrans.ToNormalization(), 
            ])
        )

    net = SegmentationNeuralNet( 
        patchproject=project, 
        nameproject=name, 
        no_cuda=cuda, 
        parallel=parallel, 
        seed=seed, 
        gpu=gpu 
        )

    if net.load( pathmodel ) is not True:
        assert(False)

    #print(len(data_loader))
    #print(net)

    #test
    idx = 0
    sample = dataset[ idx ]    
    score = net( sample['image'].unsqueeze(0) )
    score = F.resize_unet_inv_transform( score, (101,101,3), 101, cv2.INTER_LINEAR )
    #pred  = np.argmax( score, axis=2 )


    score  = sigmoid(score[:,:,0])
    pred   =  score > 0.68
    code   = rle_encode(pred)
    idname = dataset.data.getimagename( idx )

    # print( idname  )
    # print( score.shape, score.min(), score.max())
    # print( code)

    # plt.figure()
    # plt.imshow(pred)
    # plt.show()

    folder_files = os.path.expanduser('~/.datasets/tgs_salt/sample_submission.csv')
    submission = pd.read_csv( folder_files )
    results = { i:rlecode  for i,rlecode in zip( submission['id'], submission['rle_mask'] ) }    
    results[idname] = code
    results = [ {'id': k, 'rle_mask': ' '.join(map(str, v))  } for k,v in results.items()  ]
    submission = pd.DataFrame(results).astype(str)
    submission.to_csv('submission.csv', index=None, encoding='utf-8')
    
    print('DONE!!!!')



def test_submission():
    
    iname = '155410d6fa'
    folder_files = os.path.expanduser('~/.datasets/tgs_salt/sample_submission.csv')
    submission = pd.read_csv( folder_files )
    
    #submission.loc[ submission['id'] == iname , 'rle_mask'] = [1,2,3,1]
    #print( submission.loc[ submission['id'] == iname , 'rle_mask'] )
    
    results = { i:rlecode  for i,rlecode in zip( submission['id'], submission['rle_mask'] ) }
    results[iname] = [1,2,3,4, 0,0,0, 1,1,1]
    results = [ {'id': k, 'rle_mask': ' '.join(map(str, v))  } for k,v in results.items()  ]
    #print(results)

    submission = pd.DataFrame(results).astype(str)
    submission.to_csv('submission.csv', index=None, encoding='utf-8')

    
# test_submission()
test_model()