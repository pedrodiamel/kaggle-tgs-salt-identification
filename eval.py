
import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn


from pytvision.transforms.aumentation import  ObjectImageMetadataTransform
from pytvision.transforms import transforms as mtrans

from torchlib.datasets import tgsdata
from torchlib.datasets.tgsdata import TGSDataset
from torchlib.segneuralnet import SegmentationNeuralNet
from torchlib.transforms import functional as F
from torchlib.utility import rle_encode, sigmoid
from torchlib.metrics import intersection_over_union, intersection_over_union_thresholds


from argparse import ArgumentParser


def arg_parser():
    """Arg parser"""    
    parser = ArgumentParser()
    parser.add_argument('--project',     metavar='DIR',  help='path to projects')
    parser.add_argument('--projectname', metavar='DIR',  help='name projects')
    parser.add_argument('--pathdataset', metavar='DIR',  help='path to dataset')
    parser.add_argument('--namedataset', metavar='S',    help='name to dataset')
    parser.add_argument('--pathnameout', metavar='DIR',  help='path to out dataset')
    parser.add_argument('--filename',    metavar='S',    help='name of the file output')
    parser.add_argument('--model',       metavar='S',    help='filename model')  
    return parser



if __name__ == '__main__':
    
    parser = arg_parser();
    args = parser.parse_args();

    # Configuration
    project         = args.project
    projectname     = args.projectname
    pathnamedataset = os.path.join( args.pathdataset, args.namedataset )
    pathnamemodel   = args.model
    pathnameout     = args.pathnameout
    filename        = args.filename
    
    cuda=False
    parallel=False
    gpu=0
    seed=1
    imsize=101


    # Load dataset
    print('>> Load dataset ...')

    dataset  = TGSDataset(  
        pathnamedataset, 
        'train', 
        num_channels=3,
        train=True, 
        files='train.csv',
        transform=transforms.Compose([
            mtrans.ToResizeUNetFoV(imsize, cv2.BORDER_REFLECT_101),
            mtrans.ToTensor(),
            mtrans.ToNormalization(), 
            ])
        )

    # load model
    print('>> Load model ...')

    net = SegmentationNeuralNet( 
        patchproject=project, 
        nameproject=projectname, 
        no_cuda=cuda, 
        parallel=parallel, 
        seed=seed, 
        gpu=gpu 
        )

    if net.load( pathnamemodel ) is not True:
        assert(False)

    y_pred = []
    y_true = []

    for idx in tqdm( range( len(dataset) ) ):  
        
        sample = dataset[ idx ]    
        mask = sample['label'][1,:,:].data.numpy()
        mask = mask[92:92+116, 92:92+116]
        
        idname = dataset.data.getimagename( idx )
        score = net( sample['image'].unsqueeze(0) )
        
        score = F.resize_unet_inv_transform( score, (101,101,3), 101, cv2.INTER_LINEAR )
        mask  = F.resize_unet_inv_transform( mask , (101,101,3), 101, cv2.INTER_LINEAR )
    
        pred  = np.argmax( score, axis=2 )
        #pred  = sigmoid( score[:,:,0] ) > 0.5 

        y_true.append( mask.astype(int) )
        y_pred.append( pred.astype(int) )
        
        
    y_true = np.stack( y_true, axis=0 ) 
    y_pred = np.stack( y_pred, axis=0 )
        
    iout = intersection_over_union_thresholds( y_true, y_pred )

    print('IOUT:', iout )

    print('dir: {}'.format(filename))
    print('DONE!!!')



