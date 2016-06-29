# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:49:41 2016

@author: phyrct
"""

from kermit import KerrList
import os
from timeit import timeit
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\phyrct\Dropbox\Me\Coding\kermit_alpha\kermit\tests\coretestdata\hysteresis_test')
print 'starting list'
k=KerrList('image*[1-9].png', field_only=True, conserve_memory=False) #the subtract images
print 'loading images'
list(k) #load all images into memory, this takes some time

print 'starting hys'
h=k.hysteresis()
h2=k.hysteresis(box=(0,20,0,20))
h3=k.hysteresis(fieldlist=k.slice_metadata(key='Field', values_only=True))

plt.plot(h[:,0],h[:,1])
plt.plot(h2[:,0],h2[:,1])


#print k.slice_metadata(key='Field',values_only=True)

    
#print timeit('for i in range(len(k)): metadata=k[i].metadata',setup='from kermit import KerrList; k=KerrList("image*[1-9].png", field_only=True,conserve_memory=False);list(k)', number=3)
