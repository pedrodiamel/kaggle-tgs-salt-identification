



import numpy as np
import cv2
import math


def resize_unet_inv_transform(img, imgsize, fov_size, interpolate_mode): 
    
    image = np.copy(img)
    height, width = imgsize[:2]
    h,w = img.shape[:2]

    #unet required input size
    downsampleFactor = 16
    d4a_size   = 0
    padInput   = (((d4a_size *2 +2 +2)*2 +2 +2)*2 +2 +2)*2 +2 +2
    padOutput  = ((((d4a_size -2 -2)*2-2 -2)*2-2 -2)*2-2 -2)*2-2 -2    
    d4a_size   = math.ceil( (fov_size - padOutput)/downsampleFactor)
    input_size  = downsampleFactor*d4a_size + padInput
    output_size = downsampleFactor*d4a_size + padOutput;
    
    if height < width:
        asp = float(height)/width
        W = output_size
        H = int(W*asp)
    else:
        asp = float(width)/height 
        H = output_size
        W = int(H*asp)

    aspx = W/width
    aspy = H/height
        
    # inv resize mantaining aspect ratio
    image = cv2.resize(image, ( int(w*(1/aspx)) , int(h*(1/aspy) )  ) , interpolation = interpolate_mode)
    # image = cunsqueeze(image)


    return image
