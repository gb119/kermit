# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 20:46:50 2016

@author: phyrct
"""
import numpy as np
from kermit import KerrList, KerrArray

def hysteresis(images, fieldlist=None, box=None):
    """Make a hysteresis loop of the average intensity in the given images
    
    Parameters
    ----------
    ims: Kerrlist
        list of images for extracting hysteresis
    fieldlist: list or tuple 
        list of fields used, if None it will try to get field from imgae metadata
    box: list 
        [xmin,xmax,ymin,ymax] region of interest for hysteresis
    
    Returns
    -------
    hyst: nx2 np.ndarray
        fields, intensities, 2 column numpy array            
    """
    if not isinstance(images,KerrList):
        raise TypeError('images must be a KerrList')
    hys_length=len(images)    
    if fieldlist is None:
        fieldlist=images.slice_metadata(key='Field', values_only=True)
        print 'Field list: {}'.format(fieldlist)
    fieldlist=np.array(fieldlist)
    assert len(fieldlist)==hys_length, 'images and field list must be of equal length'
    assert all([i.shape==images[0].shape for i in images]), 'images must all be same shape'
    if box is None:        
        box=(0,images[0].shape[1],0,images[0].shape[0])  
    
    hyst=np.column_stack((fieldlist,np.zeros(hys_length)))
    for i,im in enumerate(images):
        im=im.crop_image(im, box, copy=True)
        hyst[i,1] = np.average(im)
    return hyst        

def drift_loop_correct(hysloop,manual=False):
    """correct a linear drift in time on a hysteresis loop"""
    pass

def faraday_correct(hysloop,manual=False):
    """correct for the faraday effect"""
    pass

def correct_image_drift(self, ref, imlist, threshold=0.005):
    """Align images to correct for image drift."""
    pass

    
def transform_images(imlist, translation=None, rotation=None):
    """Translate or rotate image or images. 
    Translates or rotates the images in the x-y plane. Areas lost by move are cropped, and 
    areas gained are made black.
    
    Parameters
    ----------
    im: array or list
        image or list of images to be translated
    tranlations: tuple or list
        array of relative distances for translation [horizontal, vert]
        eg. [[1,3],[-5,4],[9,-8]]. Defaults to no translation
    rotations: float or list
        list of rotation angles in radians to apply to the images (rotates about top left
        corner). Defaults to 0.
    
    Returns 
    newims: list 
        The transformed images (all of the same shape as the originals)
    lims: array
        The limits of the image that have not been destroyed by translations
        [xmin,xmax,ymin,ymax] (doesn't account for rotation yet!)       
    """
    single_image=False
    if not isinstance(im, list): #been given a single image
        single_image=True
        im=[im]
        
    #now make rotation and translation into a compatible list
        
    if translation is None:
        translation=np.zeros((len(im),2))
    elif isinstance(translation[0],(float,int)): #single translation to apply to all images
        translation=[translation for i in range(len(im))]
    else:
        assert len(translation)==len(im), 'translation and im dimensions are incompatible'
        
    if rotation is None:
        rotation=np.zeros((len(im),2))
    elif isinstance(rotation[0],(float,int)):
        rotation=[rotation for i in range(len(im))]
    else:
        assert len(rotation)==len(im), 'translation and im dimensions are incompatible'
            
    #apply the transform
    
    lims=[]
    newims=[]       
    for i,t,r in zip(im,translation,rotation):
        trans=transform.SimilarityTransform(translation=t,rotation=r)
        newims.append(transform.warp(i, trans))
        #now find limits of image that have not been deleted
        
        if r==0:  #no rotation, this is easy
            if t[0]<=0:
                xmin,xmax=0,i.shape[1]-t[0]
            else:
                xmin,xmax=t[0],i.shape[1]
            if t[1]<=0:
                ymin,ymax=0,i.shape[0]-t[1]
            else:
                ymin,ymax=t[1],i.shape[0]
        lims.append([xmin,xmax,ymin,ymax])
        
        if r!=0: 
            #this is harder, do the warp again but indicate lost information with 
            #a -1. Then progressively crop image until we have a maximum area
            pass #not yet implemented!
    
    lims=np.array(lims)
    if single_image:
        newims=newims[0]
        lims=lims[0]
    
    return [newims, lims] 