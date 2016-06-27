# -*- coding: utf-8 -*-
"""
Created on Tue May 03 14:31:14 2016

@author: phyrct

kermit.py

Implements KerrArray, a class for using and manipulating two tone images such
as those produced by Kerr microscopy (with particular emphasis on the Evico
software images).

Images from the Evico software are 2D arrays of 16bit unsigned integers.

"""

import numpy as np
import os
import tempfile
import PIL #check we have python imaging library plugin
import warnings
import subprocess #calls to command line
import skimage
from copy import deepcopy
from skimage import exposure,feature,io,measure,\
                    filters,graph,util,restoration,morphology,\
                    segmentation,transform,viewer



GRAY_RANGE=(0,65535)  #2^16
IM_SIZE=(512,672) #Standard Kerr image size
AN_IM_SIZE=(554,672) #Kerr image with annotation not cropped



class KerrArray(np.ndarray):
    """Class for manipulating Kerr images from Evico software.
    It is built to be almost identical to a numpy array except for one extra
    parameter which is the metadata. This stores information about the image
    in a dictionary object for later retrieval. 
    All standard numpy functions should work as normal and casting two types
    together should yield a KerrArray type (ie. KerrArray+np.ndarray=KerrArray)
    
    In addition any function from skimage should work and return a KerrArray.
    They can be called as eg. im=im.gaussian(sigma=2). Don't include the module
    name, just the function name (ie not filters.gaussian). Also omit the first
    image argument required by skimage.
    
    For clarity it should be noted that any function will not alter the current 
    instance, it will clone it first then return the clone after performing the 
    function on it.
    
    A note on coordinate systems:
    For arrays the indexing is (row, column). However the normal way to index
    an image would be to do (horizontal, vert), which is the opposite.
    In KerrArray the coordinate system is chosen similar to skimage. y points 
    down x points right and the origin is in the top left corner of the image.
    When indexing the array therefore you need to give it (y,x) coordinates
    for (row, column).
    
     ----> x (column)
    |
    |
    v
    y (row)
    
    eg I want the 4th pixel in the horizontal direction and the 10th pixel down
    from the top I would ask for KerrArray[10,4]
    
    but if I want to translate the image 4 in the x direction and 10 in the y
    I would call im=im.translate((4,10))
    
    """
    
    #first get a dictionary of funcs available from skimage library, we'll need
    #this later
    _ski_modules=[exposure,feature,io,measure,filters,filters.rank, graph,
            util, restoration, morphology, segmentation, transform, 
            viewer]
    _ski_funcs={}
    for mod in _ski_modules: 
        _ski_funcs[mod.__name__] = (mod, dir(mod))
    
    #now initialise class
    
    def __new__(cls, image, *args, **kwargs):
        """
        Construct a KerrArray object. We're using __new__ rather than __init__
        to imitate a numpy array as close as possible.
        """
        
        if isinstance(image,(str,unicode)): #we have a filename
                image=io.imread(image)
        np.array(image) #try array on image to check it's a valid numpy type
        array_args=['dtype','copy','order','subok','ndmin'] #kwargs for array setup
        array_args={k:kwargs[k] for k in array_args if k in kwargs.keys()}
        ka = np.asarray(image, **array_args).view(cls)    
        return ka #__init__ called
    
    def __init__(self, image, metadata=None, get_metadata=True, floatdata=True, 
                                    field_only=False, **kwargs):
        """Create a KerrArray instance with metadata attribute
        
        Parameters
        ----------
        image: string or numpy array initiator
            If a filename is given it will try to load the image from memory
            Otherwise it will call np.array(image) on the object so an array or
            list is suitable
        metadata: dict
            dictionary of metadata items you would like adding to your array
        get_metadata: bool
            whether to try to get the metadata from the image
        floatdata: bool
            converts the image to float 0.0-1.0 (higher resolution than uint16 
            saved by Evico). Divides the array by the max of the array dtype.
        field_only: bool
            if getting the metadata, get field only (bit faster)
        """
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata
        if isinstance(image,(str,unicode)):
            self.metadata['filename']=image
            
        if get_metadata:
            self.get_metadata(field_only=field_only) #update metadata
        if floatdata:
            self.convert_float()
            self.metadata['floatdata']=floatdata

    def __array_finalize__(self, obj):
        """__array_finalize__ and __array_wrap__ are necessary functions when
        subclassing numpy.ndarray to fix some behaviours. See
        http://docs.scipy.org/doc/numpy-1.10.1/user/basics.subclassing.html for
        more info and examples
        """
        if obj is None: return
        self.metadata = getattr(obj, 'metadata', None)

    def __array_wrap__(self, out_arr, context=None):
        """see __array_finalize__ for info"""
        return np.ndarray.__array_wrap__(self, out_arr, context)

#==============================================================
# Propety Accessor Functions
#==============================================================r
    @property
    def clone(self):
        """return a copy of the instance"""
        return KerrArray(np.copy(self),metadata=deepcopy(self.metadata), 
                               get_metadata=False,floatdata=False)
        
    @property
    def max_box(self):
        """return the coordinate extent (xmin,xmax,ymin,ymax)"""
        return (0,self.shape[1],0,self.shape[0])
    
    @property
    def data(self):
        """alias for image[:]. Equivalence to Stoner.data behaviour"""
        return self[:]
         
        
#==============================================================
#function generator
#==============================================================    
    def __getattr__(self,name):
        """run when asking for an attribute that doesn't exist yet. It
        looks first in kermit funcs then in skimage functions for a match. If
        it finds it it returns a copy of the function that automatically adds
        the image as the first argument.
        Note that numpy is already subclassed so numpy funcs are highest on the
        heirarchy, followed by kfuncs, followed by skimage funcs
        Also note that the function named must take image array as the 
        first argument"""
        
        ret=None
        #first check kermit funcs
        import kermit.kfuncs as kfuncs
        if name in dir(kfuncs):
            workingfunc=getattr(kfuncs,name)
            ret=self._func_generator(workingfunc)
        else:
            for key in self._ski_funcs.keys():
                if name in self._ski_funcs[key][1]:
                    workingfunc=getattr(self._ski_funcs[key][0],name)
                    ret=self._func_generator(workingfunc)
                    break
        if ret is None:
            raise AttributeError('No attribute found of name {}'.format(name))
        return ret
    
        
    def _func_generator(self,workingfunc):
        """generate a function that adds self as the first argument"""
        
        def gen_func(*args, **kwargs):
            r=workingfunc(self.clone, *args, **kwargs) #send copy of self as the first arg
            if type(r)!=KerrArray: #make sure we return a KerrArray
                cl=self.clone
                cl=cl.astype(r.dtype) #if the function modified the dtype don't mess with this
                cl[:]=r 
                r=cl
            return r
        
        return gen_func
        
            
#==============================================================
#Now any other useful bits
#==============================================================  
    def box(self, xmin, xmax, ymin, ymax):
        """essentially an alias for crop but always returns a view onto 
        the array rather than a copy (ie copy=False). Useful for if you
        want to do something to a partial area of an image.
        Equivalent to im[ymin,ymax,xmin,xmax]
        (the natural extension of this is using masked arrays for arbitrary
        areas, yet to be implemented)
        
        Parameters
        ----------
        xmin int
        xmax int
        ymin int
        ymax int
        
        Returns
        -------
        view: KerrArray
            view onto the array
            
        Example
        -------
        a=KerrArray([[1,2,3],[0,1,2]])
        b=a.box(0,1,0,2)
        b[:]=b+1
        print a
        #result:
        [[2,3,3],[1,2,2]]
        """
        sub=(xmin,xmax,ymin,ymax)
        return self.crop_image(box=sub,copy=False)
            
    def convert_float(self):
        """convert the image to float between 0 and 1. Dividing by the max
        allowed value of its current dtype."""
        return self.img_as_float()
        
    def convert_int(self):
        """convert the image to uint16 (the format used by Evico)"""
        return self.img_as_uint()
        
    def crop_text(self, copy=False):
        """Crop the bottom text area from a standard Kermit image
        
        Parameters
        ----------
        copy: bool
            Whether to return a copy of the data or the original data
        
        Returns
        -------
        im: KerrArray
            cropped image           
        """
           
        assert self.shape==AN_IM_SIZE or self.shape==IM_SIZE, \
                'Need a full sized Kerr image to crop' #check it's a normal image
        crop=(0,IM_SIZE[1],0,IM_SIZE[0])
        self=self.crop_image(box=crop, copy=copy)
        
    def crop_image(self, box=None, copy=True):
        """Crop the image. 
        Crops to the box given. Returns the cropped image.
        
        Parameters
        ----------
        box: array or list of type int:  
            [xmin,xmax,ymin,ymax]
        copy: bool
            whether to return a copy of the array or a view of the original object
        
        Returns
        -------
        im: KerrArray
            cropped image
        """
        if box is None:
            box=draw_rectangle(im)
        im=self[box[2]:box[3],box[0]:box[1]] #this is a view onto the
                                                    #same memory slots as self
        if copy:
            im=im.clone  #this is now a new memory location
        return im
        
    
    
    def _parse_text(self, text, key=None):
        """Attempt to parse text which has been recognised from an image
        if key is given specific hints may be applied"""
        #print '{} before processsing: \'{}\''.format(key,data)
        
        #strip any internal white space
        text=[t.strip() for t in text.split()]
        text=''.join(text)
        
        #replace letters that look like numbers
        errors=[('s','5'),('S','5'),('O','0'),('f','/'),('::','x'),('Z','2'),
                         ('l','1'),('\xe2\x80\x997','7'),('?','7'),('I','1'),
                         ('].','1'),("'",'')]
        for item in errors:
            text=text.replace(item[0],item[1])
        
        #apply any key specific corrections
        if key in ['Field','Scalebar_length_microns']:
            try:
                text=float(text)
            except:
                pass #leave it as string
        #print '{} after processsing: \'{}\''.format(key,data)
        
        return text
    
    def _tesseract_image(self, im, key):
        """ocr image with tesseract tool. 
        im is the cropped image containing just a bit of text
        key is the metadata key we're trying to find, it may give a 
        hint for parsing the text generated."""
        
        #first set up temp files to work with
        tmpdir=tempfile.mkdtemp()
        textfile=os.path.join(tmpdir,'tmpfile.txt')
        imagefile=os.path.join(tmpdir,'tmpim.tif')
        tf=open(textfile,'w') #open a text file to export metadata to temporarily
        tf.close()
        
        #process image to make it easier to read
        i=skimage.img_as_float(im)
        i=exposure.rescale_intensity(i,in_range=(0.49,0.5)) #saturate black and white pixels
        i=exposure.rescale_intensity(i) #make sure they're black and white
        i=transform.rescale(i, 5.0) #rescale to get more pixels on text
        io.imsave(imagefile,i,plugin='pil') #python imaging library will save according to file extension
    
        #call tesseract
        try:      
            subprocess.call(['tesseract', imagefile, textfile[:-4]]) #adds '.txt' extension itself
        except:
            warnings.warn('Could not call tesseract for extracting metadata '+
                     'from images, please ensure tesseract is a valid command on your '+
                     'command line', RuntimeWarning)
        tf=open(textfile,'r')
        data=tf.readline()
        tf.close()
    
        #delete the temp files
        os.remove(textfile)
        os.remove(imagefile)
        os.rmdir(tmpdir)
        
        #parse the reading
        if len(data)==0:
            print 'No data read for {}'.format(key)
        data=self._parse_text(data, key=key)
        return data
    
    def _get_scalebar(self):
        """Get the length in pixels of the image scale bar"""
        box=(0,419,519,520) #row where scalebar exists
        im=self.crop_image(box=box, copy=True)
        im=skimage.img_as_float(im)
        im=exposure.rescale_intensity(im,in_range=(0.49,0.5)) #saturate black and white pixels
        im=exposure.rescale_intensity(im) #make sure they're black and white
        im=np.diff(im[0]) #1d numpy array, differences
        lim=[np.where(im>0.9)[0][0],
             np.where(im<-0.9)[0][0]] #first occurance of both cases
        assert len(lim)==2, 'Couldn\'t find scalebar'
        return lim[1]-lim[0]
        
    def get_metadata(self, field_only=False):
        """Use image recognition to try to pull the metadata numbers off the image
        
        Requirements: This function uses tesseract to recognise the image, therefore
        tesseract file1 file2 must be valid on your command line.
        Install tesseract from 
        https://sourceforge.net/projects/tesseract-ocr-alt/files/?source=navbar
    
        Parameters
        ----------
        field_only: bool
            only try to return a field value

        Returns
        -------
        metadata: dict
            updated metadata dictionary
        """
        if self.shape!=AN_IM_SIZE:
            pass #can't do anything without an annotated image
    
        #now we have to crop the image to the various text areas and try tesseract
        elif field_only:
            fbox=(110,165,527,540) #(This is just the number area not the unit)
            im=self.crop_image(box=fbox,copy=True)
            field=self._tesseract_image(im,'Field')
            self.metadata['Field']=field
        else:
            text_areas={'Field': (110,165,527,540),
                        'Date': (542,605,512,527),
                        'Time': (605,668,512,527),
                        'Subtract': (237,260,527,540),
                        'Average': (303,350,527,540)}
            try:
                sb_length=self._get_scalebar()
            except AssertionError:
                sb_length=None
            if sb_length is not None:
                text_areas.update({'Scalebar_length_microns': (sb_length+10,sb_length+27,514,527),
                                   'Lens': (sb_length+51,sb_length+97,514,527),
                                    'Zoom': (sb_length+107,sb_length+149,514,527)})
            
            metadata={}   #now go through and process all keys
            for key in text_areas.keys():
                im=self.crop_image(box=text_areas[key], copy=True)
                metadata[key]=self._tesseract_image(im,key)
            metadata['Scalebar_length_pixels']=sb_length
            if type(metadata['Scalebar_length_microns'])==float:
                metadata['microns_per_pixel']=metadata['Scalebar_length_microns']/sb_length
                metadata['pixels_per_micron']=1/metadata['microns_per_pixel']
                metadata['field_of_view_microns']=np.array(IM_SIZE)*metadata['microns_per_pixel']
            self.metadata.update(metadata)
        if 'Field' in self.metadata.keys() and not isinstance(self.metadata['Field'],(int,float)):
            self.metadata['Field']=np.nan  #didn't read the field properly
        return self.metadata            
        
  


