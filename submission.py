
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

    folder_files = os.path.expanduser( os.path.join(pathnamedataset, 'sample_submission.csv' )  )
    submission = pd.read_csv( folder_files )
    results = { i:rlecode  for i,rlecode in zip( submission['id'], submission['rle_mask'] ) }  

    for idx in tqdm( range( len(dataset) ) ):   
        
        sample = dataset[ idx ]    
        idname = dataset.data.getimagename( idx )
        score = net( sample['image'].unsqueeze(0) )
        score = F.resize_unet_inv_transform( score, (101,101,3), 101, cv2.INTER_LINEAR )
    
        pred  = np.argmax( score, axis=2 )        
        #pred  = sigmoid( score[:,:,0] ) > 0.5 
        pred  = pred.astype(int)

        if pred.sum() == 0 or pred.sum() > pred.shape[0]*pred.shape[1] :
            continue

        code  = rle_encode(pred)
        if len(code) == 0:
            #print('>>w: code zeros')
            continue
        
        results[idname] = code

    results = [ {'id': k, 'rle_mask': ' '.join(map(str, v))  } for k,v in results.items()  ]
    submission = pd.DataFrame(results).astype(str)
    submission.to_csv(filename, index=None, encoding='utf-8')

    print('dir: {}'.format(filename))
    print('DONE!!!')



