# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:09:04 2016

@author: phyrct
"""

from kermit import KerrArray
import numpy as np
import unittest
import sys
from os import path

#data arrays for testing - some useful small images for tests

td1=np.array(\
[[ 0.,  0.,  1.,  0.,  0.],
 [ 0.,  1.,  1.,  1.,  0.],
 [ 1.,  1.,  2.,  1.,  1.],
 [ 0.,  1.,  1.,  1.,  0.],
 [ 0.,  0.,  1.,  0.,  0.]])
 
"""
[[ 0.  0.  0.  0.  0.]
 [ 0.  1.  1.  1.  0.]
 [ 0.  1.  2.  1.  0.]
 [ 0.  1.  1.  1.  0.]
 [ 0.  0.  0.  0.  0.]]
"""


td2=np.array(\
[[ 0.,  0.,  0.,  0.,  0., 1.5],
 [ 1.,  0,  1.,  1.,  0., 0.],
 [ 1.,  0,  0.,  2.5,  1., 0],
 [ 0.,  1,  0.,  1.,  0., 0],
 [ 0.,  0.,  0.,  0.,  0., 0]])
 
"""
 [[ 0.   0.   0.   0.   0.   1.5]
  [ 1.   0.   1.   1.   0.   0. ]
  [ 1.   0.   0.   2.5  1.   0. ]
  [ 0.   1.   0.   1.   0.   0. ]
  [ 0.   0.   0.   0.   0.   0. ]]
"""
 
thisdir=sys.argv[0]
class KerrArrayTest(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(thisdir,'coretestdata')
    

    def setUp(self):
        self.td1=KerrArray(td1)
        self.td2=KerrArray(td2)
        self.anim=KerrArray('coretestdata/im1_annotated.png')
        self.unanim=KerrArray('coretestdata/im2_noannotations.png')
        self.testdata=(self.td1,self.td2,self.anim,self.unanim)
    
    def test_load(self):
        t1=KerrArray([1,2,3], metadata={'a':5},floatdata=False)
        self.assertTrue(np.array_equal(t1, np.array([1,2,3])), 'Initialising from list failed')
        self.assertTrue(t1.metadata['a']==5, 'Initialising metadata from data failed')
        t1=KerrArray([1,2,3], floatdata=False)
        self.assertTrue(np.array_equal(t1, np.array([1,2,3])))
        #done most checks here, if there was a problem loading a file it would have come up in
        #set up.
        
    def test_clone(self):
        self.td1.metadata['testclone1']='abc'
        self.td1.metadata['testclone2']=345 #add some metadata to check
        td1=self.td1.clone
        self.assertTrue(isinstance(td1,KerrArray), 'Clone not KerrArray')
        self.assertTrue(np.array_equal(td1,self.td1.clone),'Clone not replicating elements')
        self.assertTrue(all([k in td1.metadata.keys() for k in \
                    self.td1.metadata.keys()]), 'Clone not replicating metadata')
        self.assertFalse(td1.base is self.td1 or self.td1.base is td1, 'memory overlap on clone') #formal check
        td1[0,0]=10  #easier check
        self.assertTrue(td1[0,0]!=self.td1[0,0], 'memory overlap on clone (first element)')
        self.assertTrue('testclone' not in self.td1.metadata.keys(), 'problem with test func')
        td1.metadata['testclone']='abc'
        self.assertTrue('testclone' not in self.td1.metadata.keys(), 'memory overlap for metadata on clone')
    
    def test_max_box(self):
        s=self.anim.shape
        self.assertTrue(self.anim.max_box==(0,s[1],0,s[0]))
    
    def test_data(self):
        self.assertTrue(np.array_equal(self.td1.data,self.td1[:]), 'self.data doesn\'t look like the data')
        t=self.td1.clone
        t.data[0,0]+=1
        self.assertTrue(np.array_equal(t.data,t), 'self.data did not change')
    
    def test_box(self):
        t=self.td1.clone
        b=t.box(1,2,1,3)        
        self.assertTrue(b.shape==(2,1))
        b[0,0]+=1
        self.assertTrue(t[1,1]==(self.td1[1,1]+1), 
                        'box does not have same memory space as original array')
        
    def test_metadata(self):
        #this incidently tests get_metadata too
        m=self.anim.metadata
        self.assertTrue(all((m['Average']=='on,8x',
                            m['Date']=='05/02/16',
                            m['Field'] == -507.53)), 'Missing metadata')
        self.assertTrue(m['Scalebar_length_microns']==50.0)
        keys=('Scalebar_length_pixels', 'field_of_view_microns',
                          'filename', 'microns_per_pixel', 'pixels_per_micron')
        self.assertTrue(all([k in m.keys() for k in keys]), 'some part of the metadata didn\'t load')  
        m_un=self.unanim.metadata
        self.assertTrue('Field' not in m_un.keys(), 'Unannotated image has wrong metadata')                  
        self.assertTrue(isinstance(self.td1.metadata,dict), 
                            'Metadata not a dict')
        td1=self.td1.clone
        td1.metadata['testmeta']='abc'
        self.assertTrue(td1.metadata['testmeta']=='abc', 'Couldn\'t change metadata')
        del(td1.metadata['testmeta'])
        self.assertTrue('testmeta' not in td1.metadata.keys(),'Couldn\'t delete metadata')
    
    def test_crop_image(self):
        td1=self.td1.clone
        c=td1.crop_image(box=(1,3,1,4),copy=False)
        self.assertTrue(np.array_equal(c,self.td1[1:4,1:3]),'crop didn\'t work')
        self.assertTrue(c.base is td1, 'crop copied image when it shouldn\'t')
        c=td1.crop_image(box=(1,3,1,4),copy=True)
        self.assertFalse(c.base is td1, 'crop didn\'t copy image')
    
    def test_other_funcs(self):
        td1=self.td1.clone
        td1=td1.level_image() #test kfuncs
        td1=td1.img_as_float() #test skimage
        td1=td1.gaussian(sigma=2)
        
if __name__=="__main__": # Run some tests manually to allow debugging
    #test=KerrArrayTest()
    #test.setUp()
    unittest.main()
        
        
        
        