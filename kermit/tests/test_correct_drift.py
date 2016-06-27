# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:21:48 2016

@author: phyrct
"""
from kermit import KerrArray
import numpy as np

a=KerrArray('coretestdata/im2_noannotations.png')

b=a.translate((2.5,3))
c=b.correct_drift(ref=a)
print c.metadata

assert np.allclose(np.array(c.metadata['correct_drift']),np.array([-2.52,-3.0]))
