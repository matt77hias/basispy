# -*- coding: utf-8 -*-
import numpy as np
from math_utils import normalize, copysign

# Calculates an orthonormal basis from a given unit vector with the method of
# Hughes and Möller.
def OrthonormalBasis_HughesMoller(n):
    if np.abs(n[0]) > np.abs(n[2]):
        u = np.array([-n[1], n[0], 0.0], dtype=np.float32)
    else:
        u = np.array([0.0, -n[2], n[1]], dtype=np.float32)
    b2 = normalize(u)
    b1 = np.cross(b2, n)
    return (n, b1, b2)

# Calculates an orthonormal basis from a given unit vector with the method of
# Frisvad.
def OrthonormalBasis_Frisvad(n):
    if (n[2] < -0.9999999):
        b1 = np.array([ 0.0, -1.0, 0.0], dtype=np.float32)
        b2 = np.array([-1.0,  0.0, 0.0], dtype=np.float32)
        return (n, b1, b2)
        
    a = 1.0 / (1.0 + n[2])
    b = -n[0] * n[1] * a
    b1 = np.array([1.0 - n[0] * n[0] * a, b, -n[0]], dtype=np.float32)
    b2 = np.array([b, 1.0 - n[1] * n[1] * a, -n[1]], dtype=np.float32)
    return (n, b1, b2)

# Calculates an orthonormal basis from a given unit vector with the method of
# Duff, Burgess, Christensen, Hery, Kensler, Liani and Villemin.
def OrthonormalBasis_Duff(n):
    s = copysign(1.0, n[2])
    a = -1.0 / (s + n[2])
    b = n[0] * n[1] * a
    b1 = np.array([1.0 + s * n[0] * n[0] * a, s * b, -s * n[0]], dtype=np.float32)
    b2 = np.array([b, s + n[1] * n[1] * a, -n[1]], dtype=np.float32)
    return (n, b1, b2)

# Calculates the error (deviatiation from orthonormality) of a given orthonormal
# basis
def OrthonormalBasis_Error(n, b1, b2):
    o_n = np.linalg.norm(n) - 1.0
    o_b1 = np.linalg.norm(b1) - 1.0
    o_b2 = np.linalg.norm(b2) - 1.0
    n_dot_b1 = np.dot(n, b1)
    n_dot_b2 = np.dot(n, b2)
    b1_dot_b2 = np.dot(b1, b2)
    return (o_n*o_n + o_b1*o_b1 + o_b2*o_b2 + n_dot_b1*n_dot_b1 \
        + n_dot_b2*n_dot_b2 + b1_dot_b2*b1_dot_b2) / 6.0
