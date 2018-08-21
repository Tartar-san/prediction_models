import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage import measure 
from skimage import feature
from skimage.morphology import disk
from skimage.filters import rank
from skimage import exposure
import warnings
warnings.filterwarnings("ignore")


def draw(file, rgb=0):
    '''
    Draws original image, correspondin mask and sum of two'''
    plt.figure(figsize = (15,10))
    plt.subplot(131, title ='Original Image')
    plt.imshow(plt.imread(file))
    plt.axis('off')
    plt.subplot(132, title ='Mask of an Image')
    plt.imshow(masks_all(file))
    plt.axis('off')
    plt.subplot(133, title ='Combined') 
    plt.imshow(plt.imread(file)[:,:,rgb]+masks_all(file)*200)
    plt.axis('off')
    plt.suptitle(file, y=0.77, verticalalignment ='top', fontsize = 22)
    plt.show()
    print('')
    
    
def Draw_contour(file, param=150):
    '''
    Plots together contour on an original image,
    zoomed part of original image containing contoure and contoure alone
    in each of three color channels
    
    Example of usage:
    for file in np.random.choice(ships, 20):
        Draw_contour(file, Parameter(file))
    '''
    img = io.imread(file)
    img0, img1, img2 = img[:,:,0],img[:,:,1],img[:,:,2]
    contour0, contour1, contour2 = Contour(file, param)
    fig, (ax, ax3, ax2) = plt.subplots(ncols=3, figsize=(20, 10))
    fig.figsize = (20,20)
    ax.plot(contour0[::,1], contour0[::,0], color = 'g', linewidth = 0.2)
    ax.imshow(img0, origin='lower', cmap='Reds')
    ax.axis('off')
    ax.set(title = 'Red Spectrum')
    ax2.plot(contour0[::,1], contour0[::,0], color = 'tan')
    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    ax2.set_aspect(asp)
    ax2.axis('off')
    ax2.set(title = 'Only Contour')
    ax3.set(xlim=(min(contour0[::,1])-50,max(contour0[::,1])+50), ylim=(min(contour0[::,0])-50,max(contour0[::,0])+50), autoscale_on=False,
               title='Zoom Section with Contour')
    ax3.plot(contour0[::,1], contour0[::,0], color = 'g', linewidth = 0.9)
    ax3.imshow(img0, cmap='Reds')
    ax3.axis('off')
    plt.suptitle(file, y=0.85, fontsize = 22, x=0.25)
    plt.show()

    fig, (ax, ax3, ax2) = plt.subplots(ncols=3, figsize=(20, 10))
    fig.figsize = (20,20)
    ax.plot(contour1[::,1], contour1[::,0], color = 'r', linewidth = 0.2)
    ax.imshow(img1, origin='lower', cmap='Greens')
    ax.axis('off')
    ax.set(title = 'Green Spectrum')
    ax2.plot(contour1[::,1], contour1[::,0], color = 'tan')
    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    ax2.set_aspect(asp)
    ax2.axis('off')
    ax2.set(title = 'Only Contour')
    ax3.set(xlim=(min(contour1[::,1])-50,max(contour1[::,1])+50), ylim=(min(contour1[::,0])-50,max(contour1[::,0])+50), autoscale_on=False,
               title='Zoom Section with Contour')
    ax3.plot(contour1[::,1], contour1[::,0], color = 'r', linewidth = 0.9)
    ax3.imshow(img1, cmap='Greens')
    ax3.axis('off')
    plt.suptitle(file, y=0.85, fontsize = 22, x=0.25)
    plt.show()

    fig, (ax, ax3, ax2) = plt.subplots(ncols=3, figsize=(20, 10))
    fig.figsize = (20,20)
    ax.plot(contour2[::,1], contour2[::,0], color = 'r', linewidth = 0.2)
    ax.imshow(img2, origin='lower', cmap='Blues')
    ax.axis('off')
    ax.set(title = 'Blue Spectrum')
    ax2.plot(contour2[::,1], contour2[::,0], color = 'tan')
    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    ax2.set_aspect(asp)
    ax2.axis('off')
    ax2.set(title = 'Only Contour')
    ax3.set(xlim=(min(contour2[::,1])-20,max(contour2[::,1])+20), ylim=(min(contour2[::,0])-20,max(contour2[::,0])+20), autoscale_on=False,
               title='Zoom Section with Contour')
    ax3.plot(contour2[::,1], contour2[::,0], color = 'r', linewidth = 0.9)
    ax3.imshow(img2, cmap='Blues')
    ax3.axis('off')
    plt.suptitle(file, y=0.85, fontsize = 22, x=0.25)
    plt.show()
    print('')
    
    
def draw_contrast(file):
    '''
    Plots together original image, 
    high contrast image (in rgb and in each of color channels)
    and lower contrast image
    
    Example of usage:
    for file in np.random.choice(ships, 20):
        draw_contrast(file)
    '''
    img = io.imread(file) 
    print('')
    plt.figure(figsize = (15,15))
    plt.subplot(161, title ='Original')
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(162, title ='High Contrast (HC)')
    plt.imshow(Contrast(file))
    plt.axis('off')
    plt.subplot(163, title ='HC Red Spectrum') 
    plt.imshow(Contrast(file, 0), cmap='inferno')
    plt.axis('off')
    plt.subplot(164, title ='HC Green Spectrum') 
    plt.imshow(Contrast(file, 1), cmap='inferno')
    plt.axis('off')
    plt.subplot(165, title ='HC Blue Spectrum') 
    plt.imshow(Contrast(file, 2), cmap='inferno')
    plt.axis('off')
    plt.subplot(166, title ='Equalized Image') 
    plt.imshow(Equalizer(file), cmap='inferno')
    plt.axis('off')
    plt.suptitle(file, y=0.63, fontsize = 22)
    plt.show()      
    
    
def plot_all(file):
    '''
    Plots all extracted features together
    
    Example of usage:
    for file in np.random.choice(ships, 20):
        plot_all(file)
    '''
    img = io.imread(file)
    plt.figure(figsize = (12,15))
    plt.subplot(431)
    plt.xlim(4000,0)
    plt.imshow(ship)
    plt.axis('off')
    plt.subplot(432, title ='Original')
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(433)
    plt.imshow(ship)
    plt.axis('off')
    plt.subplot(434, title ='Mask')
    plt.imshow(masks_all(file))
    plt.axis('off')
    plt.subplot(435, title ='High Contrast (HC)')
    plt.imshow(Contrast(file))
    plt.axis('off')
    plt.subplot(436, title ='Equalized Image') 
    plt.imshow(Equalizer(file), cmap='inferno')
    plt.axis('off')
    plt.subplot(437, title ='HC Red Spectrum') 
    plt.imshow(Contrast(file, 0), cmap='inferno')
    plt.axis('off')
    plt.subplot(438, title ='HC Green Spectrum') 
    plt.imshow(Contrast(file, 1), cmap='inferno')
    plt.axis('off')
    plt.subplot(439, title ='HC Blue Spectrum') 
    plt.imshow(Contrast(file, 2), cmap='inferno')
    plt.axis('off')
    c0, c1, c2 = Contour(file, param=Parameter(file))
    plt.subplot(4,3,10, title ='Red Spectrum Contours') 
    plt.plot(c0[::,1], c0[::,0], color = 'tan')
    plt.axis('off')
    plt.imshow(np.zeros((768,768)), cmap = 'summer')
    plt.subplot(4,3,11, title ='Green Spectrum Contours') 
    plt.plot(c0[::,1], c0[::,0], color = 'tan')
    plt.axis('off')
    plt.imshow(np.zeros((768,768)), cmap = 'summer')
    plt.subplot(4,3,12, title ='Blue Spectrum Contours') 
    plt.plot(c0[::,1], c0[::,0], color = 'tan')
    plt.axis('off')
    plt.imshow(np.zeros((768,768)), cmap ='summer')
    plt.suptitle(file, y=0.92, fontsize = 22)
    plt.show()
    print('')