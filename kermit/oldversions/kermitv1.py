# -*- coding: utf-8 -*-
"""
Created on Tue May 03 14:31:14 2016

@author: phyrct

kerr_util_funcs.py

A few useful funcs for manipulating Kerr images

It mostly assumes it's taking standard Kerr microscopy png images produced
by the provided Evico software. These are 16bit unsigned integer intensity
images, .
"""

import numpy as np
import os, sys, time
import tempfile
from os import path
#from copy import copy
import PIL #check we have python imaging library plugin
import warnings
#from matplotlib.pyplot import imshow
import subprocess #calls to command line
import skimage
from skimage import draw,exposure,feature,io,measure,\
                    filters,util,restoration,segmentation,\
                    transform
#from skimage.viewer import ImageViewer,CollectionViewer


GRAY_RANGE=(0,65535)  #2^16
IM_SIZE=(512,672)
AN_IM_SIZE=(554,672) #Kerr image with annotation not cropped

StringTypes=(str,unicode)

ex_data1='ExampleData1'
ex_data2='ExampleData2'
tmp_dir='tmp'


class KerrArray(np.ndarray):
    """Class for manipulating Kerr images from Evico software.
    It is built to be almost identical to a numpy array except for one extra
    parameter which is the metadata. This stores information about the image
    in a dictionary object for later retrieval. 
    All standard numpy functions should work as normal and casting two types
    together should yield a KerrArray type (ie. KerrArray+np.ndarray=KerrArray)
    
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
    
    """
    
    def __new__(cls, image, metadata={}, get_metadata=True, datatype='float'):
        """
        Construct a Kermit object. We're using __new__ rather than __init__
        to imitate a numpy array as close as possible.
        
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
        datatype: 'float' or 'int'
            converts the image to float 0.0-1.0 or uint8 0-65535
        Returns
        -------
        ka: KerrArray
            A KerrArray object with metadata attached
        """
        if isinstance(image,(str,unicode)): #we have a filename
                metadata['filename']=image
                image=io.imread(image)
        np.array(image) #try array on image to check it's a valid numpy type
        ka = np.asarray(image).view(cls)    
        ka.__init__(image, metadata=metadata,get_metadata=get_metadata,
                        datatype=datatype)
        return ka
    
    def __init__(self, image, metadata={}, get_metadata=True, datatype='float'):
        """called by __new__ when it has finished"""
        
        #so we call __init__ in __new__ but we don't want to run again in the
        #second call at the end of the __new__ func so here's a flag to stop it
        if hasattr(self, 'initrun'):           
            pass
        else:
            self.initrun=True
        
        #now a normal init
            self.metadata = metadata
            if get_metadata:
                self.get_metadata() #update metadata
            if datatype=='float':
                fl=skimage.img_as_float(self)
                #self=self.astype(np.float64)
                #self[:]=fl
                #print self
            elif datatype=='int':
                self[:]=skimage.img_as_uint(self)
            else:
                raise ValueError('Datatype must be \'float\' or \'int\'')


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
# Property Accessor Functions
#==============================================================
    @property
    def clone(self):
        return KerrArray(np.copy(self),metadata=self.metadata, 
                               get_metadata=False)
        
    @property
    def max_box(self):
        return (0,self.shape[1],0,self.shape[0]) #(xmin,xmax,ymin,ymax)
        
#==============================================================
#initialisation stuff completed, now onto the interesting bit
#==============================================================    

    def convert_float(self):
        """convert the image to float between 0 and 1"""
        fl=skimage.img_as_float(self)
        self=self.astype(np.float64)
        self[:]=fl
        return self
        
    def convert_int(self):
        """convert the image to uint8 (the format used by Evico)"""
        i=skimage.img_as_uint(self)
        self=self.astype(np.int)
        self[:]=i
        return self
        
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
        im=self.crop_image(box=crop, copy=copy)
        return im
        
    def crop_image(self, box=None, copy=True):
        """Crop the image. 
        Crops to the box given or defaults to allowing the user
        to draw a rectangle. Returns the cropped image.
        
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
        
        
    def level_image(self, poly_vert=1, poly_horiz=1, box=None, poly=None):
        """Subtract a polynomial background from image
        
        Fit and subtract a background to the image. Fits a polynomial of order
        given in the horizontal and vertical directions and subtracts. If box 
        is defined then level the *entire* image according to the 
        gradient within the box.
        
        Parameters
        ----------
        poly_vert: int
            fit a polynomial in the vertical direction for the image of order 
            given. If 0 do not fit or subtract in the vertical direction
        poly_horiz: int
            fit a polynomial of order poly_horiz to the image. If 0 given
            do not subtract
        box: array, list or tuple of int
            [xmin,xmax,ymin,ymax] define region for fitting. IF None use entire
            image
        poly: list or None
            [pvert, phoriz] pvert and phoriz are arrays of polynomial coefficients
            (highest power first) to subtract in the horizontal and vertical 
            directions. If None function defaults to fitting its own polynomial.
            
        Returns
        -------
        im: KerrArray
            the levelled image
        """
        if box is None:
            box=self.max_box
        cim=self.crop_image(box=box)
        (vertl,horizl)=cim.shape
        if poly_horiz>0:
            comp_vert = np.average(cim, axis=0) #average (compress) the vertical values
            if poly is not None:
                p=poly[0]
            else:
                p=np.polyfit(np.arange(horizl),comp_vert,poly_horiz) #fit to the horizontal
                av=np.average(comp_vert) #get the average pixel height
                p[-1]=p[-1]-av #maintain the average image height
            horizcoord=np.indices(self.shape)[1] #now apply level to whole image 
            for i,c in enumerate(p):
                self=self-c*horizcoord**(len(p)-i-1)
            self.metadata['poly_vert_subtract']=p
        if poly_vert>0:
            comp_horiz = np.average(cim, axis=1) #average the horizontal values
            if poly is not None:
                p=poly[1]
            else:
                p=np.polyfit(np.arange(vertl),comp_horiz,poly_vert)
                av=np.avearage(comp_horiz)
                p[-1]=p[-1]-av #maintain the average image height
            vertcoord=np.indices(self.shape)[0]
            for i,c in enumerate(p):
                self=self-c*vertcoord**(len(p)-i-1)
            self.metadata['poly_horiz_subtract']=p
        return self
    
    
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
        #os.remove(tmpdir)
        
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
        return self.metadata            
                
    def line_profile(self, start, end, width=1, order=1):
        """Return a line trace of intensity averaging over width.
        Call through to skimage.measure.profile_line
        
        Parameters
        ----------
        start: 2-tuple
            coords at start of line (numpy coordinates not x,y)
        end: 2-tuple
            coords at end of line (last pixel is included in result)
        width: int
            width of line to average over
        order: int
            order of the spline interpolation
        
        Returns
        -------
        profile: array
            intensity profile
        """
        return measure.profile_line(self,src=start,dst=end,linewidth=width,
                                        order=order,mode='nearest')       
    
    def filter_image(self, sigma=2, box=None):
        """Apply a filter to an area of the image defined by box
        call through to skimage.filters.gaussian
        
        Parameters
        ----------
        sigma: float
            standard deviation for gaussian blur
        box: 4-tuple
            area to apply blur to (xmin,xmax,ymin,ymax)
        
        Returns
        -------
        image
            filtered image
        """
        if box==None:
            box=(0,self.shape[1],0,self.shape[0])
        im=self.crop_image(box=box, copy=True)
        im=filters.gaussian(im, sigma=sigma)
        self[box[2]:box[3],box[0]:box[1]]=im
        return self
    

        
    
    def scale_intensity(self, lims=(0.1,0.9), percent=True):
        """rescale the intensity of the image
        
        Parameters
        ----------
        lims: 2-tuple
            limits of rescaling the intensity
        percent: bool
            if True then lims are the give the percentile of the image intensity
            histogram
        
        Returns
        -------
        image: KerrArray
            rescaled image
        """
        if percent:
            vmin,vmax=np.percentile(self,np.array(lims)*100)
            print vmin, vmax
        else:
            vmin,vmax=lims[0],lims[1]
        ret=exposure.rescale_intensity(self,in_range=(vmin,vmax)) #clip the intensity 
        self[:]=ret
        return self
        
        
        
    def translate(self, translation):
        """Translates the image.
        Areas lost by move are cropped, and areas gained are made black (0)
        
        Parameters
        ----------
        translate: 2-tuple
            translation (x,y)
        
        Returns
        -------
        im: KerrArray
            translated image
        """
        trans=transform.SimilarityTransform(translation=translation)
        self[:]=transform.warp(self, trans)
        return self
        
    
    def rotate(self, rotation):
        """Rotates the image.
        Areas lost by move are cropped, and areas gained are made black (0)
        
        Parameters
        ----------
        rotation: float
            clockwise rotation angle in radians (rotated about top right corner)
        
        Returns
        -------
        im: KerrArray
            rotated image
        """
        rot=transform.SimilarityTransform(rotation=rotation)
        self=transform.warp(self, rot)
        return self
        
    
    def translate_limits(self, translation):
        """Find the limits of an image after a translation
        After using KerrArray.translate some areas will be black,
        this finds the area that still has original pixels in
        
        Parameters
        ----------
        translation: 2 tuple
            the (x,y) translation applied to the image
        
        Returns
        -------
        limits: 4-tuple
            (xmin,xmax,ymin,ymax"""
        t=translation
        s=self.shape
        if t[0]<=0:
            xmin,xmax=0,s[1]-t[0]
        else:
            xmin,xmax=t[0],s[1]
        if t[1]<=0:
            ymin,ymax=0,s[0]-t[1]
        else:
            ymin,ymax=t[1],s[0]
        return (xmin,xmax,ymin,ymax)

    def correct_drift(self, ref, threshold=0.005):
        """Align images to correct for image drift.
        Detects common features on the images and tracks them moving.
        
        Parameters
        ----------
        ref: KerrArray or ndarray
            reference image with zero drift
        threshold: float
            threshold for detecting imperfections in images 
            (see skimage.feature.corner_fast for details)
        
        Returns
        -------
        shift: array
            shift vector relative to ref (x drift, y drift)
        transim: KerrArray
            copy of self translated to account for drift"""
        refed=ref.clone
        refed=filters.gaussian(ref,sigma=1)
        refed=feature.corner_fast(refed,threshold=0.005)
        imed=self.clone
        imed=filters.gaussian(imed,sigma=1)
        imco=feature.corner_fast(imed,threshold=0.005)
        shift,err,phase=feature.register_translation(refed,imco,upsample_factor=50)
        #tform = SimilarityTransform(translation=(-shift[1],-shift[0]))
        #imed = transform.warp(im, tform) #back to original image
        self=self.translate(translation=(-shift[1],-shift[0]))
        return [shift,self]   

        
    def split_image(self):
        """split image into different domains, maybe by peak fitting the histogram?"""
        pass

    
    def edge_det(filename,threshold1,threshold2):
        '''Detects an edges in an image according to the thresholds 1 and 2.
        Below threshold 1, a pixel is disregarded from the edge
        Above threshold 2, pixels contribute to the edge
        Inbetween 1&2, if the pixel is connected to similar pixels then the pixel conributes to the edge '''
        pass
     
    def NPPixel_BW(np_image,thresh1,thresh2):
        '''Changes the colour if pixels in a np array according to an inputted threshold'''
        pass
        
    

    
class kskimage(skimage):
    
        
    
    def skifunc(self, func, args, box=None, **kwargs):
    """Apply an skimage function to the KerrArray. 
    This keeps the KerrArray type rather than a normal skimage function 
    which will usually return a numpy array, wiping the metadata.
    """



class KerrGUI():
    def _rect_selected(extents):
        """event function for skimage.viewer.canvastools.RectangleTool
        """
        rect_coord.update({'done': True})
         
    def draw_rectangle(self):
        """Draw a rectangle on the image and return the coordinates
        
        Returns
        -------
        box: ndarray
            [xmin,xmax,ymin,ymax]"""
            
        viewer=ImageViewer(self)
        viewer.show()
        from skimage.viewer.canvastools import RectangleTool
        rect_selected_yet={'done':False} #mutable object to store func status in
        rect_tool = RectangleTool(viewer, on_enter=_rect_selected)
        while not rect_selected_yet['done']:
            time.sleep(2)
            pass
        coords=np.int64(rect_tool.extents)
        viewer.close()
        return coords 
    
    def draw_trace(vert_coord, width=1):
        """Line trace horizontal at vertical coord averaging over width"""
        pass
    
    def plt_histogram(self, **kwarg):
        """plot histogram of image intensities, pass through kwarg to matplotlib.pyplot.hist"""
        plt.hist(self.ravel(), **kwarg)
    


if __name__=='__main__':
    example_im_fol=r'C:\Users\phyrct\Dropbox\Me\Coding\kermit'
    #os.chdir(example_im_fol)
    #os.chdir(ex_data2)
#    bkim=io.imread('bknd.png')
#    unpim=io.imread('unpro.png')
#    im=io.imread('sub.png')
#    proc_list=[im]
#    proc_list.append(crop_text(im)) #crop annotation text from image 
#    v1=CollectionViewer(proc_list)     
#    v1.show()    


