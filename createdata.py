
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
    
    index = [
        1335,1109,3824,1044,3029,3859,1131,1388,1888,3555,1187,551,225,1105,381,298,2195,264,178,3366,2435,813,260,3009,794,
        1357,2849,3820,1050,1545,1424,308,3661,151,1909,2800,717,3035,3391,1987,3247,1074,169,26,2897,2211,2892,401,253,1200,
        2035,635,3011,3137,3334,2545,727,1612,514,2620,196,3951,2671,4,3267,1464,3420,3832,1068,3274,886,315,21,1717,202,3791,
        12,236,2819,1261,110,2456,2297,3984,1473,3685,250,3768,1754,2859,116,1041,1005,2281,2665,3809,1747,2650,3785,2535,2374,
        238,3180,2658,1697,786,1997,1627,1022,168,1400,3587,1869,905,3696,3624,773,2082,2331,2041,2104,2289,1998,3781,1597,111,
        348,259,3365,898,2181,3335,95,656,1777,336,2496,3250,910,1477,929,1583,3498,3232,1558,3122,311,744,2229,3790,3633,2830,
        2404,130,1805,1530,851,3112,2048,2737,3967,3140,1861,2694,3038,2789,3157,2786,1192,1092,667,1982,2765,3436,2891,3752,2426,
        1569,1953,3491,3870,2683,3618,2559,3064,1066,2328,3023,2925
        ]
    

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
            'bool': True,  # mg > 0.1 and (area==0 or area > 10) not i in index,
            }  )
        
    metadata = pd.DataFrame( metadata )
    print(metadata.head())
    print(metadata['bool'].sum()) 
    

    filename=os.path.join(pathnamedataset,'metadata_train.csv')
    metadata.to_csv(filename, index=None, encoding='utf-8')
    print('save train metadata ...')
    
    
    return

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