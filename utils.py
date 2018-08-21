import os
import numpy as np
import pandas as pd
from skimage import io
from skimage import measure 
from skimage import feature
from skimage.morphology import disk
from skimage.filters import rank
from skimage import exposure
import warnings
warnings.filterwarnings("ignore")


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def mask_part(pic):
    '''
    Function that converts single marker from 'marks' into the image
    '''
    back = np.zeros(Img_Height**2)
    starts = pic.split()[0::2]
    lens = pic.split()[1::2]
    for i in range(len(lens)):
        back[(int(starts[i])-1):(int(starts[i])-1+int(lens[i]))] = 1
    return np.reshape(back, (Img_Height, Img_Width))


def is_empty(key):
    '''
    Checks if there is a marker for specific image
    i.e. if there is a ship on image
    '''
    df = marks[marks['ImageId'] == key].iloc[:,1]
    if len(df) == 1 and type(df.iloc[0]) != str and np.isnan(df.iloc[0]):
        return True
    else:
        return False


def masks_all(key):
    '''
    Collects together all single markers belonging to the same image
    '''
    df = marks[marks['ImageId'] == key].iloc[:,1]
    masks= np.zeros((Img_Height, Img_Width))
    if is_empty(key):
        return masks
    else:
        for i in range(len(df)):
            masks += mask_part(df.iloc[i])
        return masks.T
    
    
def Parameter(file):
    p = np.arange(10,250,20)
    for i in p:
        if Param(file, i):
            break
    return i


def Param(file, param):
    '''
    Chooses feasible parameter for the function Contour (below)'''
    img = io.imread(file)
    img0 = img[:,:,0]
    contours0 = measure.find_contours(img0, param)
    img1 = img[:,:,1]
    contours1 = measure.find_contours(img1, param)
    img2 = img[:,:,2]
    contours2 = measure.find_contours(img2, param)
    if len(contours0) == 0 or len(contours1)==0 or len(contours2)==0:
        return False
    else:
        return True
    
    
def Contour(file, param=150):
    '''
    Looks for contours on images across all three channels
    '''
    img = io.imread(file)
    img0 = img[:,:,0]
    contours0 = measure.find_contours(img0, param)
    contour0 = max(contours0, key=len)
    img1 = img[:,:,1]
    contours1 = measure.find_contours(img1, param)
    contour1 = max(contours1, key=len)
    img2 = img[:,:,2]
    contours2 = measure.find_contours(img2, param)
    contour2 = max(contours2, key=len)
    return contour0, contour1, contour2


# Equalizing and Stretching Image Contrast
def Equalizer(file):
    '''
    Equalizes image spectrum (in gray scale)
    '''
    img = io.imread(file, as_grey=True)
    selem = disk(100)
    img_eq = rank.equalize(img, selem=selem)
    return img_eq