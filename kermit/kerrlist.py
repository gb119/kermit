# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:05:59 2016

@author: phyrct
"""

from kermit import KerrArray
import numpy as np
import os, sys, time
from os import path
from copy import copy
import PIL #check we have python imaging library plugin
import skimage
from skimage.io import ImageCollection
from skimage.io import imread
from skimage import filters, feature
#from skimage import draw,exposure,feature,io,measure,\
#                    filters,util,restoration,segmentation,\
#                    transform
#from skimage.viewer import ImageViewer,CollectionViewer


GRAY_RANGE=(0,65535)  #2^16
IM_SIZE=(512,672)
AN_IM_SIZE=(554,672) #Kerr image with annotation not cropped

StringTypes=(str,unicode)


def _load_KerrArray(f,img_num=0, **kwargs):
    return KerrArray(f, **kwargs)    

class KerrList(ImageCollection):
    """KerrList groups functions that can be applied to a group of KerrImages.
    In general it is designed to behave pretty much like a normal python list.
    """
    
    def __init__(self, load, conserve_memory=True, 
                     load_func=None, **load_func_kwargs):
        """
        Initialise a KerrList. A list of images to manipulate. Mostly a pass
        through to the skimage.io.ImageCollection class
        
        Parameters
        ----------
        load: str or list
            pattern of filenames to load or list of arrays or KerrArrays. Uses 
            standard glob nomenclature. Seperate
            different requests with a : eg 'myimages/*.png:myimages/*.jpg'. If
            a list is given it treats it as a list of image arrays.
        conserve_memory: bool
            parameter passed onto skimage.io.ImageCollection. Option for loading
            all the files into memory initially or later
        
        Other parameters
        ----------------
        load_func: callable or None
            see skimage.io.ImageCollection for notes.
        
        Attributes
        ----------
        files: list or str
            list of loaded file names. Or equal to load_pattern if a list was
            given.
        
        """
        if load_func is None:
            load_func=_load_KerrArray
        if isinstance(load,KerrList):
            load=list(load) #load all arrays and return as list, bit of 
                                #a hack because we lose filenames etc. from original
                                #could be more careful
        super(KerrList, self).__init__(load, conserve_memory=conserve_memory, 
                    load_func=load_func, **load_func_kwargs)
        self.altdata=[None for i in self.files] #altdata for if arrays are changed
                                                #from those loaded from memory
           
    
    def __setitem__(self, n, array):
        """Set item at index n to array given.
        
        ImageCollection was originally written to deal with only files, it
        loads them on demand. Set item was not included because after that the
        load function doesn't work. However this limits the use of collection
        because we can't manipulate the files we have added unless we add them
        to a list.
        """
        current=self[n] #this loads item from memory if it isn't already plus
                        #has checks for a valid n
        
        if not isinstance(array, np.ndarray):
            raise ValueError('Set item must be numpy array')
        self.altdata[n]=array
        
    def __getitem__(self, n):
        """Get item at index n.        
        """ 
        ret = super(KerrList,self).__getitem__(n) #gets value from self.data
        if isinstance(n,slice):
            return ret #another imagecollection
        idx = n % len(self.altdata)
        if not self.altdata[idx] is None:
            ret=self.altdata[idx]
        return ret
    
    def __getattr__(self,name):
        """run when asking for an attribute that doesn't exist yet. It
        looks in listfuncs for a match. If
        it finds it it returns a copy of the function that automatically adds
        the KerrList as the first argument."""
        
        ret=None
        import kermit.listfuncs as listfuncs
        if name in dir(listfuncs):
            workingfunc=getattr(listfuncs,name)
            ret=self._func_generator(workingfunc)
        if ret is None:
            raise AttributeError('No attribute found of name {}'.format(name))
        return ret
    
        
    def _func_generator(self,workingfunc):
        """generate a function that adds self as the first argument"""
        
        def gen_func(*args, **kwargs):
            r=workingfunc(self, *args, **kwargs) #send copy of self as the first arg
            return r
        
        return gen_func
        
    def __delitem__(self, n):
        """For deleting and appending items we load all items and ignore the
        filenames. Then we can treat it more like a list
        """
        l=list(self)
        del(l[n])
        self.__init__(l)
    
    def all_arrays(self):
        """Load all files and return a KerrList with only arrays in it. Better
        if we're going to append and delete items"""
        self.__init__(list(self))
        
    def append(self, ap, index=None):
        """append an array to the list. All items will be loaded and filenames
        will be lost.
        ap: KerrArray or np.ndarray
            item to append
        index: index to insert item at"""
        if not isinstance(ap, np.ndarray):
            raise TypeError('Appended item must be an array')
        l=list(self)
        if index==None:
            index=len(l) #add to the end
        l.insert(index,l)
        self.__init__(l)
        
    def slice_metadata(self, key=None, values_only=False):
        """Return a list of the metadata dictionaries for each item/file
        
        Parameters
        ----------
        key: string or list of strings
            if given then only return the item(s) requested from the metadata
        values_only: bool
            if given only return tuples of the dictionary values. Mostly useful
            when given a single key string
        Returns
        ------
        ret: list of dict, tuple or values
            depending on values_only returns the sliced dictionaries or tuples/
            values of the items
        """
        print 'loading self'
        self.all_arrays()
        print 'loading metadata'
        metadata=[k.metadata for k in self]
        if isinstance(key, (str,unicode)):
            key=[key]
        if isinstance(key, list):
            for i,met in enumerate(metadata):
                print 'assert statement'
                assert all([k in met for k in key]), 'key requested not in item {}'.format(i)
                print 'generating metadata'
                metadata[i]={k:v for k,v in metadata[i].iteritems() if k in key}
        if values_only:
            for i,met in enumerate(metadata):
                metadata[i]=[v for k,v in met.iteritems()]
            if len(metadata[0])==1: #single key
                metadata=[m[0] for m in metadata]
        return metadata

        
