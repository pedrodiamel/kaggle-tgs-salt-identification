


import numpy as np
import pytest
import cv2
import matplotlib.pyplot as plt 

import sys
sys.path.append('../')
from torchlib.transforms import functional as F
from pytvision.transforms import functional as FF

def test_transfom():
    
    img = np.random.randn( 116,116,3 )
    #img = FF.resize_unet_transform( img, 101, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT )
    image = F.resize_unet_inv_transform( img, (101,101,3), 101, cv2.INTER_LINEAR )
    
    print(img.shape)
    print(image.shape)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(image)
    plt.show()


test_transfom()