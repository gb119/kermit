# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:45:26 2016

@author: phyrct
"""

import os
import numpy as np
from kermit import KerrArray, KerrList
from skimage.viewer import CollectionViewer,ImageViewer
import matplotlib.pyplot as plt

folder=r'..\tests\coretestdata\hysteresis_test'
editfolder=r'..\tests\coretestdata\hysteresis_test\edited_ims'

os.chdir(folder)
background=KerrArray('image0000_background.png')
background=background.convert_float()
print 'loading images...'
kl=KerrList('image000*unproccessed.png', conserve_memory=False, get_metadata=False)
list(kl) #make sure all images loaded
print 'images loaded'



#%%

print 'convert float'
kl.apply_all('convert_float') #convert pixels to decimal values
print 'correct_drift'
kl.apply_all('correct_drift', ref=background, upsample_factor=200)
print 'subtract'
for i,im in enumerate(kl):
    kl[i]=16*(kl[i]-background)+0.5
    kl[i]=kl[i].clip_intensity()
    #skimage.io.imsave(os.path.join(savefolder,kl.files[i]),kl[i])

print np.array(kl.slice_metadata(key='correct_drift', values_only=True))

cv=CollectionViewer(kl)
cv.show()