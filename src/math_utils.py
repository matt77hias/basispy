# -*- coding: utf-8 -*-
import numpy as np

# Normalizes the given vector.
def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0.0 else v/norm
   
# Copy the sign of the second argument to
# the first argument (sign(0.0) = +)
def copysign(v, x):
    return v if x >= 0.0 else -v