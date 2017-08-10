# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from onb import OrthonormalBasis_HughesMoller, OrthonormalBasis_Frisvad, OrthonormalBasis_Duff, OrthonormalBasis_Error

class OrthonormalBasisStrategy:
    
    def __init__(self, name, f):
        self.name = name
        self.f = f
        
    def OrthonormalBasis(self, n):
        return self.f(n)
        
    def Start(self, res):
        self.data = np.zeros([res, res], dtype=np.float32)
        self.count = np.zeros([res, res], dtype=np.uint32)
        self.worst_error = 0.0
        self.worst_basis = None
        
    def AddSample(self, py, px, n):
        basis = self.OrthonormalBasis(n)
        error = OrthonormalBasis_Error(*basis)
        if self.worst_error < error:
            self.worst_error = error
            self.worst_basis = basis
        self.data[py,px] += error
        self.count[py,px] += 1
        
    def Stop(self):
        for py in range(self.data.shape[0]): 
            for px in range(self.data.shape[1]):
                if self.count[py,px] != 0:
                    self.data[py,px] /= self.count[py,px]
        return self.data
  
strategies = [
     OrthonormalBasisStrategy('HughesMoller', OrthonormalBasis_HughesMoller), \
     OrthonormalBasisStrategy('Frisvad', OrthonormalBasis_Frisvad), \
     OrthonormalBasisStrategy('Duff', OrthonormalBasis_Duff) \
     ]

def test(nb_samples=16, d=0.001, rng=np.random):
    
    res = int(2.0 / d + 1.0)
    for strategy in strategies:
        strategy.Start(res)

    y = -1.0
    for py in range(res):
        x = -1.0
        for px in range(res):
            for i in range(nb_samples):
                nx = x + rng.uniform() * d
                ny = y + rng.uniform() * d
                nz2 = 1.0 - nx*nx - ny*ny
                if nz2 < 0.0:
                    continue
                nz = -np.sqrt(nz2)
                n = np.array([nx, ny, nz], dtype=np.float32)
                
                for strategy in strategies:
                    strategy.AddSample(py, px, n)
            
            x += d  
        y +=d
         
    for strategy in strategies:
        data = strategy.Stop()
        
        print(strategy.name + ': ' + str(strategy.worst_error))
        
        plt.figure(strategy.name)
        plt.title(strategy.name)
        plt.imshow(data, cmap='jet', interpolation='none', norm=colors.LogNorm(vmin=10**-16, vmax=1.0))
        plt.colorbar()
        plt.axis('off')