# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:49:41 2016

@author: phyrct
"""

from kermit import KerrList
import os

os.chdir(r'C:\Users\phyrct\Dropbox\Me\Coding\kermit_alpha\kermit\tests\coretestdata\hysteresis_test')
print 'starting list'
k=KerrList('image*000[1-9].png', field_only=True) #the subtract images

print 'starting hys'
#h=k.hysteresis()

#print h

print k[0]