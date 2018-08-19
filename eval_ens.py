
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

    dataset = TGSDataset(  
        pathnamedataset, 
        'train', 
        num_channels=3,
        train=True, 
        files='train.csv',
        transform=transforms.Compose([
            mtrans.ToResize( (256,256), resize_mode='squash', padding_mode=cv2.BORDER_REFLECT_101 ),
            #mtrans.ToResizeUNetFoV(imsize, cv2.BORDER_REFLECT_101), #unet
            mtrans.ToTensor(),
            #mtrans.ToNormalization(), 
            mtrans.ToMeanNormalization( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], )
            ])
        )

    # load model
    print('>> Load model ...')

    # /$PROJECTNAME/$PATHMODEL/$NAMEMODEL  
    dataprojects = [
        ['exp_tgs_unetresnet_152_mcedice_adam_tgs-salt-identification-challenge_001', 'chk000335.pth.tar'],
        ['exp_tgs_unetresnet_152_mcedice_adam_tgs-salt-identification-challenge_001', 'chk000510.pth.tar'],
        ['exp_tgs_unetresnet_mcedice_adam_tgs-salt-identification-challenge_002', 'chk000210.pth.tar'],
        ['exp_tgs_unet11_mcedice_adam_tgs-salt-identification-challenge_001', 'chk000340.pth.tar'],        
    ] 
    
    nets = []
    for projectname, model in dataprojects:
        net = SegmentationNeuralNet( 
            patchproject=project, 
            nameproject=os.path.join(project, projectname), 
            no_cuda=cuda, 
            parallel=parallel, 
            seed=seed, 
            gpu=gpu 
            )

        if net.load( os.path.join(project, projectname, 'models', model) ) is not True:
            assert(False)
        
        nets.append(net)

    y_pred = []
    y_true = []
    

    index = []
    tta = False
    for idx in tqdm( range( len(dataset) ) ):  #len(dataset)
        
        sample = dataset[ idx ]    
        mask = sample['label'][1,:,:].data.numpy()
        #mask = mask[92:92+116, 92:92+116] #unet
        
        idname = dataset.getimagename( idx )
        #score = net( sample['image'].unsqueeze(0).cuda(), sample['metadata'].unsqueeze(0).cuda() )   
        image  = sample['image'].unsqueeze(0)
        image  = image.cuda()
        
        if (image-image.min()).sum() == 0:
            continue
        
        scores = []
        for net in nets:
            score = net( image, sample['metadata'].unsqueeze(0).cuda() )
            if tta:
                score_t = net( F.fliplr( image ), sample['metadata'].unsqueeze(0).cuda() )
                score   = score + F.fliplr( score_t )
                score_t = net( F.flipud( image ), sample['metadata'].unsqueeze(0).cuda() )
                score   = score + F.flipud( score_t )
                score_t = net( F.flipud( F.fliplr( image ) ), sample['metadata'].unsqueeze(0).cuda() )
                score   = score + F.flipud( F.fliplr( score_t ) )
                score = score/4
            
            scores.append( score )
        
        scores = torch.stack(scores,dim=0)
        ## mean ruler
        score = score.mean(dim=0)
        ## to numpy
        score = score.data.cpu().numpy().transpose(1,2,0)
        
                    
        #score = F.resize_unet_inv_transform( score, (101,101,3), 101, cv2.INTER_CUBIC )  #unet
        #mask  = F.resize_unet_inv_transform( mask , (101,101,3), 101, cv2.INTER_LINEAR ) #unet
        
        pred  = np.argmax( score, axis=2 )
        #pred  =  score[:,:,1]  > 0.40 #sigmoid()
        #pred = tgspostprocess(score)
        
        index.append( pred.sum() > 5 )

        y_true.append( mask.astype(int) )
        y_pred.append( pred.astype(int) )
        
    
    index  = np.stack( index , axis=0 )
    y_true = np.stack( y_true, axis=0 ) 
    y_pred = np.stack( y_pred, axis=0 )
        
    iout = intersection_over_union_thresholds( y_true[index], y_pred[index] )
    iou  = intersection_over_union( y_true[index], y_pred[index] )

    print('IOUT:', iout )
    print('OUT:', iout )

    print('dir: {}'.format(filename))
    print('DONE!!!')


