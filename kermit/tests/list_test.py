# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:21:48 2016

@author: phyrct
"""
from kermit import KerrArray,KerrList
import numpy as np

a=KerrList('coretestdata/*.png')


#check loaded ok
print a[1].metadata
assert all([k in ['floatdata','filename'] for k in a[1].metadata.keys()])

#check set item
a[0]=KerrArray([1,2,3],dtype=np.float64,metadata={'abc':3})
assert 'abc' in a[0].metadata.keys()
print a[0].metadata
a.files

#check reload doesn't destroy setitems
a.reload() 
print a[0].metadata
assert 'abc' in a[0].metadata.keys()


#check loading from arrays
print 'load array test'
b=KerrList([KerrArray('coretestdata\\im1_annotated.png'), 
            KerrArray('coretestdata\\im2_noannotations.png')])
print b[1].metadata
assert isinstance(b[1],np.ndarray)
assert isinstance(b.files[1],np.ndarray)

#check loading from KerrList
c=KerrList(a)
d=KerrList(b)
print c[1].metadata
assert 'floatdata' in c[0].metadata.keys()
assert isinstance(c[1],np.ndarray)
assert isinstance(c.files[1],np.ndarray)
assert 'floatdata' in d[0].metadata.keys()
assert isinstance(d[1],np.ndarray)
assert isinstance(d.files[1],np.ndarray)

#check slice metadata
c.slice_metadata()
print c.slice_metadata(key='floatdata',values_only=True)

"""
#check append
c.append(KerrArray([2,3,4]))
print len(c)
print c[2]
"""
