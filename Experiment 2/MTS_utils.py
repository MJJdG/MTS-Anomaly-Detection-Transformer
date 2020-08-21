#!/usr/bin/env python
# coding: utf-8
"""
Collection of functions and classes used by the models
@author: Max de Grauw (M.degrauw@student.ru.nl)
"""

import numpy as np

class Scaler3DMM_tanh():

    def __init__(self,X,y=None):
        self.maxs, self.mins = [], []
        for signal in X.T:
            self.maxs.append(np.amax(signal))
            self.mins.append(np.amin(signal))
        self.scalef = max(self.maxs)/(self.maxs)
            
    def transform(self,X):
        Xt = np.zeros(X.shape)
        for idx, signal in enumerate(X.T):
            Xt.T[idx] = (2*(signal - self.mins[idx]) / (self.maxs[idx] - self.mins[idx]))-1
        return Xt
    
    def inverse_transform(self,X):
        Xi = np.zeros(X.shape)
        for idx, signal in enumerate(X.T):
            Xi.T[idx] = (((signal+1)/2) * (self.maxs[idx] - self.mins[idx]) + self.mins[idx])
        return Xi

class Scaler3DMM():

    def __init__(self,X,y=None):
        self.maxs, self.mins = [], []
        for signal in X.T:
            self.maxs.append(np.amax(signal))
            self.mins.append(np.amin(signal))
        self.scalef = max(self.maxs)/(self.maxs)
        
    def transform(self,X):
        Xt = np.zeros(X.shape)
        for idx, signal in enumerate(X.T):
            Xt.T[idx] = (signal - self.mins[idx]) / (self.maxs[idx] - self.mins[idx])
        return Xt
    
    def inverse_transform(self,X):
        Xi = np.zeros(X.shape)
        for idx, signal in enumerate(X.T):
            Xi.T[idx] = signal * (self.maxs[idx] - self.mins[idx]) + self.mins[idx]
        return Xi

class Signal(object):
    def __init__(self, Amp, Freq, L, d, Fs):
        self.Amp = Amp
        self.Freq = Freq
        self.d = d
        self.Fs = Fs
        self.t = np.linspace(0, L, L*Fs)
                
        for cls in reversed(self.__class__.mro()):
            if hasattr(cls, 'init'):
                cls.init(self)

    def setL(self, L, Fs):
        self.t = np.linspace(0, L, L*Fs)

class Sine(Signal):
    def __init__(self, Amp=1, Freq=.5, L=100, d=.1, Fs=1):
        super().__init__(Amp, Freq, L, d, Fs)

    def getSignal(self):
        return self.Amp * (np.sin(self.Freq * 2 * np.pi * self.t) +
                np.random.normal(scale=self.d, size=self.t.size))
    
class Cosine(Signal):
    def __init__(Signal, Amp=1, Freq=.5, L=100, d=.1, Fs=1):
        super().__init__(Amp, Freq, L, d, Fs)
    
    def getSignal(self):
        return self.Amp * (np.cos(self.Freq * 2 * np.pi * self.t) + 
                np.random.normal(scale=self.d, size=self.t.size))
