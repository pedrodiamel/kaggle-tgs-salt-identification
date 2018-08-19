
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
from torchlib.postprocessing import tgspostprocess


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
        metadata='metadata_test.csv',
        filter=False,
        transform=transforms.Compose([
            mtrans.ToResize( (256,256), resize_mode='squash', padding_mode=cv2.BORDER_REFLECT_101 ),
            #mtrans.ToResizeUNetFoV(imsize, cv2.BORDER_REFLECT_101),
            mtrans.ToTensor(),
            #mtrans.ToNormalization(), 
            mtrans.ToMeanNormalization( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], )
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

    #folder_files = os.path.expanduser( os.path.join(pathnamedataset, 'sample_submission.csv' )  )
    #submission = pd.read_csv( folder_files )
    #results = { i:rlecode  for i,rlecode in zip( submission['id'], submission['rle_mask'] ) }  
    
    tta = True
    results = list()    
    for idx in tqdm( range( len(dataset) ) ):   
        
        sample = dataset[ idx ]    
        idname = dataset.data.getimagename( idx )
        image  = sample['image'].unsqueeze(0)
        
        if (image-image.min()).sum() == 0:
            results.append( {'id':idname, 'rle_mask':' '  } )
            continue
        
        score = net( image.cuda(), sample['metadata'][1].unsqueeze(0).unsqueeze(0).cuda() )
        if tta:
            score_t = net( F.fliplr( image.cuda() ), sample['metadata'][1].unsqueeze(0).unsqueeze(0).cuda() )
            score   = score + F.fliplr( score_t )
            score_t = net( F.flipud( image.cuda() ), sample['metadata'][1].unsqueeze(0).unsqueeze(0).cuda() )
            score   = score + F.flipud( score_t )    
            score_t = net( F.flipud( F.fliplr( image.cuda() ) ), sample['metadata'][1].unsqueeze(0).unsqueeze(0).cuda() )
            score   = score + F.flipud( F.fliplr( score_t ) )
            score = score/4
            
        score = score.data.cpu().numpy().transpose(2,3,1,0)[...,0]
        
        #score = F.resize_unet_inv_transform( score, (101,101,3), 101, cv2.INTER_CUBIC ) #unet
        #score = cv2.resize(score, (101, 101) , interpolation = cv2.INTER_CUBIC) #unet
    
        pred  = np.argmax( score, axis=2 )          
        #pred  = sigmoid( score[:,:,0] ) > 0.5
        #pred = tgspostprocess(score)
        
        pred  = cv2.resize(pred.astype(float), (101, 101) , interpolation=cv2.INTER_LINEAR)  
        pred  = pred.astype(int)


        code  = rle_encode(pred)
        code  = ' '.join( map(str, code) )

        if pred.sum() <= 10: # area <= 10 
            code = ' '        

        if len(code) == 0:
            #print('>>w: code zeros')
            code = ' '
        
        #results[idname] = code
        results.append( {'id':idname, 'rle_mask':code  } )
        

    #results = [ {'id': str(k), 'rle_mask': ' '.join( map(str, v) )  } for k,v in results.items()  ]    
    submission = pd.DataFrame(results).astype(str)
    submission.to_csv(filename, index=None, encoding='utf-8')

    print('dir: {}'.format(filename))
    print('DONE!!!')



