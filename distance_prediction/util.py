import skimage.io
import skimage.transform
from PIL import ImageFile
import os
import ipdb
import scipy.misc as misc
import numpy as np

#def load_image( path, height=128, width=128 ):
def load_image( path, pre_height=128, pre_width=128, height=128, width=128 ):

    try:
        img = skimage.io.imread( path ).astype( float )
    except:
        return None

    img /= 255.

    if img is None: return None
    if len(img.shape) < 2: return None
    if len(img.shape) == 4: return None
    if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
    if img.shape[2] == 4: img=img[:,:,:3]
    if img.shape[2] > 4: return None

    short_edge = min( img.shape[:2] )
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy+short_edge, xx:xx+short_edge]
    resized_img = skimage.transform.resize( crop_img, [pre_height,pre_width] )

    #rand_y = np.random.randint(0, pre_height - height)
    #rand_x = np.random.randint(0, pre_width - width)

    #resized_img = resized_img[ rand_y:rand_y+height, rand_x:rand_x+width, : ]

    return (resized_img * 2)-1 #(resized_img - 127.5)/127.5

def load_chain( path, pre_length=128 ):

    try:
        chain = np.loadtxt( path )
    except:
        return None
    #temp=chain.split()
    chain_1=chain[0]
    chain_2=chain[1]
    a=chain_1+chain_2    
    b=chain_1*pre_length
    d=b//a
    c=int(d)
    return c

def crop_random(image_ori, width=64,height=64, x=None, y=None, overlap=7):
    if image_ori is None: return None
    random_y = np.random.randint(overlap,height-overlap) if x is None else x
    random_x = np.random.randint(overlap,width-overlap) if y is None else y

    image = image_ori.copy()
    crop = image_ori.copy()
    crop = crop[random_y:random_y+height, random_x:random_x+width]
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 0] = 2*117. / 255. - 1.
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 1] = 2*104. / 255. - 1.
    image[random_y + overlap:random_y+height - overlap, random_x + overlap:random_x+width - overlap, 2] = 2*123. / 255. - 1.

    return image, crop, random_x, random_y

def crop_interaction(image_ori, c):
    if image_ori is None: return None
#    random_y = np.random.randint(overlap,height-overlap) if x is None else x
#    random_x = np.random.randint(overlap,width-overlap) if y is None else y
    x=c
    y=c
    image = image_ori.copy()
    crop_temp = image_ori.copy()
    crop_temp2 = crop_temp[0:y, x:128]
    image[0:y, x:128, 0] = 2*117. / 255. - 1.
    image[0:y, x:128, 1] = 2*104. / 255. - 1.
    image[0:y, x:128, 2] = 2*123. / 255. - 1.
    crop_1=misc.imresize(crop_temp2[:,:,0],[64,64],interp='nearest')
    crop_2=misc.imresize(crop_temp2[:,:,1],[64,64],interp='nearest')
    crop_3=misc.imresize(crop_temp2[:,:,2],[64,64],interp='nearest')
#    crop_temp3=crop_temp
    crop=crop_temp[0:64,64:128]
    crop[:,:,0]=crop_1
    crop[:,:,1]=crop_2
    crop[:,:,2]=crop_3
    return image, crop,x,y
