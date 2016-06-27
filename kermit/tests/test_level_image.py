# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:21:48 2016

@author: phyrct
"""
from kermit import KerrArray
import numpy as np

a=KerrArray([[.1,.2,.3],
             [.2,.3,.4],
             [.3,.4,.5]], dtype=np.float64)

b=a.level_image()
c=a.level_image(poly_vert=0)
assert np.allclose(b,np.array([[.3,.3,.3],[.3,.3,.3],[.3,.3,.3]],dtype=np.float64))

c=a.level_image(poly_vert=0)
assert np.allclose(c,np.array([[.2,.2,.2],[.3,.3,.3],[.4,.4,.4]],dtype=np.float64))

d=b.level_image(poly=-np.array(b.metadata['poly_sub']))
assert np.allclose(d,a) #should get it back to original

