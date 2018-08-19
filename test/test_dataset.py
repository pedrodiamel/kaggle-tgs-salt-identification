
import sys
sys.path.append('../')
from torchlib.datasets import TGSDataset
from torchlib.datasets import weightmaps
from torchlib.datasets.imageutl import TGSProvide, TGSExProvide
 

import matplotlib.pyplot as plt 
import numpy as np
import pytest

def summary( tensor ):
    print( tensor.shape, tensor.max(), tensor.min() )

def test_tgs_provide():
    
    path  = '~/.kaggle/competitions/tgs-salt-identification-challenge/'
    #data  = TGSProvide.create(  path, sub_folder='test', train=False, files='sample_submission.csv' )
    #name, image, depth = data[ np.random.randint( len(data) ) ]

    data  = TGSProvide.create(  path, sub_folder='train', train=True, files='train.csv' )
    name, image, mask, depth = data[ np.random.randint( len(data) ) ]
    

    print(name)
    summary(image)
    summary(mask)
    summary(depth)

    plt.figure()
    plt.subplot(131)
    plt.imshow(image)
    plt.subplot(132)
    plt.imshow(mask)
    plt.subplot(133)
    plt.imshow( weightmaps.getweightmap(mask) )
    plt.show()

def test_tgs_ext_provide():
    
    path  = '~/.kaggle/competitions/tgs-salt-identification-challenge/'
    #data  = TGSProvide.create(  path, sub_folder='test', train=False, files='sample_submission.csv' )
    #name, image, depth = data[ np.random.randint( len(data) ) ]

    data  = TGSExProvide.create(  path, sub_folder='train', train=True, files='train.csv', metadata='metadata_train.csv', filter=True )
    name, image, mask, depth = data[ np.random.randint( len(data) ) ]
    

    print(name)
    summary(image)
    summary(mask)
    summary(depth)

    plt.figure()
    plt.subplot(131)
    plt.imshow(image)
    plt.subplot(132)
    plt.imshow(mask)
    plt.subplot(133)
    plt.imshow( weightmaps.getweightmap(mask) )
    plt.show()

def test_tgs_train_dataset():       

    path  = '~/.kaggle/competitions/tgs-salt-identification-challenge/'
    data  = TGSDataset(  path, 'train', files='train.csv' )
    sample = data[ np.random.randint( len(data) ) ]

    #print(sample)
    image, mask, weight, depth = sample['image'], sample['label'], sample['weight'], sample['metadata']

    summary(image)
    summary(mask)
    summary(weight)
    summary(depth)

    plt.figure()
    plt.subplot(131)
    plt.imshow(image )
    plt.subplot(132)
    plt.imshow(mask[:,:,0])
    plt.subplot(133)
    plt.imshow( weight[:,:,0] )
    plt.show()

def test_tgs_test_dataset():       

    path  = '~/.kaggle/competitions/tgs-salt-identification-challenge/'
    data  = TGSDataset(  path, 'test', train=False, files='sample_submission.csv' )
    sample = data[ np.random.randint( len(data) ) ]

    #print(sample)
    image, (name, depth) = sample['image'], sample['metadata']

    print(name)
    print(depth)
    summary(image)

    plt.figure()
    plt.imshow(image )
    plt.show()


# test_tgs_provide()
# test_tgs_train_dataset()
# test_tgs_test_dataset()
test_tgs_ext_provide()