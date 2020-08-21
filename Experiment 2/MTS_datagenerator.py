# -*- coding: utf-8 -*-
"""
Synthetic Data Generator for Multivariate Time Series (MTS) Data
@author: Max de Grauw (M.degrauw@student.ru.nl)
"""

import numpy as np
from itertools import chain 
import matplotlib.pyplot as plt
import h5py
from MTS_utils import Cosine, Sine

def generationPipeline(L, L2, filename, seed=42, Fs=1, sign=20, train_size=2**13,\
                       val_size=2**12, test_size=2**12, anomaly_size=2**12):
    """
    Generates normal and anomalous synthetic MTS data using DataGenerator class
    
    Parameters:
        L: Training Sequence Length
        L2: Forecasting Window Length
        filename: output file name
        seed: Random seed to use
        Fs: Sampling rate of used signals
        sign: Number of output signals 20 or 40
        train_size: Training set size for forecasting training
        val_size: Validation ,, ,,
        test_size: Test ,, ,,
        anomaly_size = Size of dataset for anomaly prediction
    
    Returns:
        h5py file containing synthetic data
    """    
    np.random.seed(seed)
    if sign != 20 and sign != 40:
        print(sign,"is a nonstandard number of signals, requires manual changes to "+\
              "fixed parameters in data generation script. See code comments for help")
        return False
    
    ##########################################################################
    # Create datagenerator
    ##########################################################################
    dg = DataGenerator(L, L2, Fs)
    print("Generating data for",sign,"signals, with length of",L)

    ##########################################################################
    # Input & output parameter matrices:
    # 0:    number of copies, 
    # 1:    number of wave combinations, 
    # 2,0:  [random error ranges ((a->b)/c), 
    # 2,1:  random frequency ranges, 
    # 2,2:  random amplitude ranges], 
    # 3:    whether to use sine and/or cos waves
    ##########################################################################
    inputSigs =  [[1, 3, [[100, 200, 10000], [150,  200, 100000], [500, 1000, 1000]], [1, 1]], # Program
                  [1, 3, [[100, 250, 10000], [200,  250, 100000], [150,  200, 1000]], [1, 1]], # Temperature
                  [3, 5, [[250, 500, 10000], [500, 1000, 100000], [100,  150, 1000]], [1, 1]], # Vibration
                  [1, 4, [[100, 250, 10000], [250,  300, 100000], [300,  350, 1000]], [1, 1]], # Speed
                  [1, 4, [[100, 250, 10000], [250,  300, 100000], [250,  300, 1000]], [1, 1]]] # Power
    
    sigfact = 2 if sign == 40 else 1 # Double all sensors if sign == 40
    outputSigs = [[sigfact*3, 4, [[100, 250, 10000], [400, 500, 1000000], [300, 350, 1000]], [1, 1]], #GTS
                  [sigfact*2, 3, [[100, 250, 10000], [100, 200, 1000000], [150, 200, 1000]], [1, 1]], #MV
                  [sigfact*1, 5, [[100, 250, 10000], [500, 600, 1000000], [100, 150, 1000]], [1, 1]], #GTRVX
                  [sigfact*1, 5, [[100, 250, 10000], [500, 600, 1000000], [100, 150, 1000]], [1, 1]], #GTRVY
                  [sigfact*1, 5, [[100, 250, 10000], [500, 600, 1000000], [100, 150, 1000]], [1, 1]], #GBV
                  [sigfact*1, 5, [[100, 250, 10000], [500, 600, 1000000], [100, 150, 1000]], [1, 1]], #BHV
                  [sigfact*1, 5, [[100, 250, 10000], [500, 600, 1000000], [100, 150, 1000]], [1, 1]], #GTV
                  [sigfact*2, 3, [[100, 250, 10000], [200, 300, 1000000], [250, 300, 1000]], [1, 1]], #TB1
                  [sigfact*2, 3, [[100, 250, 10000], [200, 300, 1000000], [250, 300, 1000]], [1, 1]], #TB2
                  [sigfact*2, 3, [[100, 250, 10000], [200, 300, 1000000], [250, 300, 1000]], [1, 1]], #TB3
                  [sigfact*2, 3, [[100, 250, 10000], [200, 300, 1000000], [250, 300, 1000]], [1, 1]], #TB4
                  [sigfact*1, 3, [[100, 250, 10000], [200, 300, 1000000], [200, 250, 1000]], [1, 1]], #TE
                  [sigfact*1, 4, [[100, 250, 10000], [300, 400, 1000000], [250, 300, 1000]], [1, 1]]] #GTP

    outputNames = np.array(['GTS', 'MV', 'GTRVX', 'GTRVY', 'GBV', 'BHV',\
                            'GTV', 'TB1', 'TB2', 'TB3', 'TB4', 'TE', 'GTP','PROGRAM'])
    outputNames = [n.encode("ascii", "ignore") for n in outputNames]

    ##########################################################################
    # Output/input dependencies matrix (size= inputSigs*outputSigs)
    ##########################################################################
    outputDeps = [[1, 0, 0, 1, 0], #GTS
                  [1, 0, 0, 0, 0], #MV
                  [1, 0, 1, 0, 0], #GTRVX
                  [1, 0, 1, 0, 0], #GTRVY
                  [1, 0, 1, 0, 0], #GBV
                  [1, 0, 1, 0, 0], #BHV
                  [1, 0, 1, 0, 0], #GTV
                  [1, 1, 0, 0, 0], #TB1
                  [1, 1, 0, 0, 0], #TB2
                  [1, 1, 0, 0, 0], #TB3
                  [1, 1, 0, 0, 0], #TB4
                  [1, 1, 0, 0, 0], #TE 
                  [1, 0, 0, 1, 1]] #GTP

    ##########################################################################
    # Internal dependency matrix (size= outputSigs*outputSigs):
    # Signal X_t can only depend on signals t-x ... t-1
    ##########################################################################
                    #GTS    MV      GTRVX   GTRVY   GBV     BHV     GTV     TB1     TB2     TB3     TB4    TE      GTP
    internalDeps = [[0,     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0], #GTS
                    [0,     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0], #MV
                    [0.5,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0], #GTRVX
                    [0.5,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0], #GTRVY
                    [0.25,  0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0], #GBV
                    [0.25,  0,      0,      0,   0.25,      0,      0,      0,      0,      0,      0,      0,      0], #BHV
                    [0.25,  0,   0.25,   0.25,  0.125,  0.125,      0,      0,      0,      0,      0,      0,      0], #GTV
                    [0,     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0], #TB1
                    [0,     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0], #TB2
                    [0,     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0], #TB3
                    [0,     0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0], #TB4
                    [0,     0,      0,      0,      0,      0,      0,      0.25,   0.25,   0.25,   0.25,   0,      0], #TE
                    [0.25,  0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0]] #GTP

    ##########################################################################
    # Initialize data generator and plot test signals
    ##########################################################################
    dg.setParams(inputSigs, outputSigs, outputDeps, internalDeps, verbose=False)
    
    for i in range(4):
        dg.generate(verbose=True)
        
    ##########################################################################
    # Create training data
    ##########################################################################    
    x_train = np.zeros((train_size, L*Fs, sign+1))
    x_val= np.zeros((val_size, L*Fs, sign+1))
    
    y_train = np.zeros((train_size, L2*Fs, sign+1))
    y_val = np.zeros((val_size, L2*Fs, sign+1))
    
    for i in range(0, train_size+val_size):
        if i < train_size:
            x_train[i,:,:], y_train[i,:,:] = dg.generate()
        elif i < train_size+val_size:
            x_val[i-train_size,:,:], y_val[i-train_size,:,:] = dg.generate()

    print("Finished creating training dataset",train_size+val_size,"of samples")   

    ##########################################################################
    # Create testing data
    ##########################################################################
    x_test = np.zeros((test_size, L*Fs, sign+1))
    y_test = np.zeros((test_size, L2*Fs, sign+1))   
    
    for i in range(0, test_size):
        x_test[i,:,:], y_test[i,:,:] = dg.generate()
        
    print("Finished creating test dataset",test_size,"of samples")
        
    ##########################################################################
    # Anomaly parameters:
    # 0: Overall anomaly rate
    # 1: Individual anomaly rates (size = sign)
    # 2: Anomaly time ranges
    # 3: Anomaly change ranges
    # 4: Probabilities for different anomaly types
    # 5: Minimal length of anomaly in percentage of total window (L1->L1+L2) 
    # 6: Minimal value to start at when smoothing edges of anomaly
    ##########################################################################
    if sign == 20: # Base number of sensors 
        # GTS, MV, GTRVX, GTRVY, GBV, BHV, GTV, TB1, TB2, TB3, TB4, TE, GTP
        anomRates = [0.2, 0.2, 0.2,\
                     0.05, 0.05,\
                     0.1,\
                     0.1,\
                     0.1,\
                     0.1,\
                     0.1,\
                     0.1, 0.1,\
                     0.1, 0.1,\
                     0.1, 0.1,\
                     0.1, 0.1,\
                     0.1,\
                     0.2]
    if sign == 40: # Double number of sensors
        anomRates = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2,\
                     0.05, 0.05, 0.05, 0.05,\
                     0.1, 0.1,\
                     0.1, 0.1,\
                     0.1, 0.1,\
                     0.1, 0.1,\
                     0.1, 0.1,\
                     0.1, 0.1, 0.1, 0.1,\
                     0.1, 0.1, 0.1, 0.1,\
                     0.1, 0.1, 0.1, 0.1,\
                     0.1, 0.1, 0.1, 0.1,\
                     0.1, 0.1,\
                     0.2, 0.2]
            
    anomaly_params = [.9, anomRates, [L, L+L2], [0.5, 1.5], [[0.5, 0.5], [0.5, 0.5]], 0.2, 0.5]
   
    ##########################################################################    
    # Create Anomaly Data 
    ##########################################################################
    x_anomalies = np.zeros((anomaly_size, (L+L2)*Fs, sign+1))
    y_anomalies = np.zeros((anomaly_size, (L+L2)*Fs, sign+1))

    # Generate anomalous data
    for i in range(0, anomaly_size):
        if i < 8:
            x_anomalies[i,:,:], y_anomalies[i,:,:] = \
                dg.generateAnomalous(anomaly_params, verbose=True)
        else:
            x_anomalies[i,:,:], y_anomalies[i,:,:] = \
                dg.generateAnomalous(anomaly_params, verbose=False)
    print("Finished creating anomaly dataset of",anomaly_size,"samples")
    
    ##########################################################################    
    # Save Data
    ##########################################################################
    with h5py.File(filename+"_L1_"+str(L)+"_L2_"+str(L2)+"_SIG_"+str(sign)+"_SEED_"+str(seed)+".hdf5", "w") as f:
        f.create_dataset("x_train", data=x_train)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("x_val", data=x_val)
        f.create_dataset("y_val", data=y_val)
        f.create_dataset("x_test", data=x_test)
        f.create_dataset("y_test", data=y_test)
        f.create_dataset("x_anomalous", data=x_anomalies)
        f.create_dataset("y_anomalous", data=y_anomalies)
        f.create_dataset("column_names", data=outputNames)
        print("Finished saving: "+filename+"_L1_"+str(L)+"_L2_"+str(L2)+"_SIG_"+str(sign)+"_SEED_"+str(seed)+".hdf5")
        
    return dg

class DataGenerator(object):
    def __init__(self, L, L2, Fs):
        self.L = L+L2
        self.L2 = L2
        self.Fs = Fs
    
    def setParams(self, inputSigs, outputSigs, outputDeps, internalDeps, verbose=False):
        self.outputOps = []
        self.outputDeps = []
        self.internalDeps = []
        self.inputSigs = inputSigs
        self.outputSigs = outputSigs
        
        # Pick random operations for internal signal dependencies
        # Currently only use +, - here but could also add /, *- etc.
        self.internalDepsOps = [[np.random.choice(['+','-']) for i in internalDeps] \
                                for i in internalDeps]
        
        # Pick random operations for input/ouput signal dependencies
        for sigGroup in outputDeps:
            self.outputOps.append([np.random.choice(['*']) for i in sigGroup])
        
        # Add copies to output/input dependencies
        for idx, sig in enumerate(outputSigs):
            for copy in range(sig[0]): 
                self.outputDeps.append(outputDeps[idx])
                self.internalDeps.append(internalDeps[idx])

        # Fix output function in place
        self.outputs, self.outputSigOps = self.processInputs(outputSigs)
        
        if verbose:
            print("outputs:")
            print(self.outputs)
            print(self.outputSigOps)
            print("internal dependency operations")
            print(self.internalDepsOps)
        
    def processInputs(self, sigs):
        signalList, opsList = [], []
        for sig in sigs:
            signalCopies, signalOps = [], []
            
            for ld in range(sig[1]): # Currently only use +, - here
                signalOps.append(np.random.choice(['+','-']))
                
            signal = None
            combSigs = [] 
            if sig[-1][0] == False: # Cosine
                signal = Cosine
            elif sig[-1][1] == False: # Sine
                signal = Sine
            else: # Combinations
                for ld in range(sig[1]): # Set combinations by group
                    combSigs.append(np.random.choice([Sine, Cosine]))
            
            opsList.append(signalOps)
            fs, amps, ds = [], [], []
            for ld in range(sig[1]): # Set combinations by group
                fs.append(np.random.randint(sig[2][1][0], sig[2][1][1])/sig[2][1][2])
                amps.append(np.random.randint(sig[2][2][0], sig[2][2][1])/sig[2][2][2])
                ds.append(np.random.randint(sig[2][0][0], sig[2][0][1])/sig[2][0][2])

            for copy in range(sig[0]): 
                lindeps = []
                for ld in range(sig[1]):
                    if not combSigs: # Combinations
                        lindeps.append(signal(Amp=amps[ld], Freq=fs[ld],
                                              L=self.L, d=ds[ld], 
                                              Fs=self.Fs))
                    else: 
                        lindeps.append(combSigs[ld](Amp=amps[ld], Freq=fs[ld],
                                                    L=self.L, d=ds[ld], 
                                                    Fs=self.Fs))
                        
                signalCopies.append(lindeps)
            signalList.append(signalCopies)
            
        return signalList, opsList
        
    def generate(self, scale=1e2, verbose=False): 
        # Generate new input functions
        self.inputs, self.inputSigOps = self.processInputs(self.inputSigs) 
        inps = []
        for i, _ in enumerate(self.inputs):
            inpg = []
            start = np.random.choice([-scale, scale])
            for signal in self.inputs[i]:
                j = 0
                for component in signal:
                    if j == 0:
                        inp = start*component.getSignal()
                        j += 1
                    else:
                        inp = eval("inp"+self.inputSigOps[i][j]+
                                       "component.getSignal()")
                        j += 1
                inpg.append(inp)
            inps.append(inpg)
            
        outs = []
        for i, _ in enumerate(self.outputs):
            outg = []
            for signal in self.outputs[i]:
                j = -1
                # Create signal from components
                for component in signal:
                    if j == -1:
                        out = component.getSignal()
                        j += 1
                    else:
                        out = eval("out"+self.outputSigOps[i][j]+
                                   "component.getSignal()")
                
                # Resolve dependencies on input
                for k, _ in enumerate(self.outputDeps[i]):
                    if self.outputDeps[i][k]:
                        out = eval("out"+self.outputOps[i][k]+
                                   "np.mean(inps[k], axis=0)")
                
                # Resolve internal dependencies
                for k, _ in enumerate(self.internalDeps[i]):
                    if self.internalDeps[i][k] > 0:
                        out = eval("out"+self.internalDepsOps[i][k]+
                                   "(self.internalDeps[i][k]*\
                                       np.mean(outs[k], axis=0))")
                outg.append(out)
            outs.append(outg)
            
        input_sequence = np.array(list(chain.from_iterable(inps))).T
        output_sequence = np.array(list(chain.from_iterable(outs))).T

        if verbose:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7));
            ax1.plot(input_sequence);
            ax1.set_title('Input Signals');
            ax2.plot(output_sequence);
            ax2.set_title('Output Signals');
            
        # Add program signal to dataset    
        full_output_sequence = np.zeros((output_sequence.shape[0], output_sequence.shape[1]+1))
        full_output_sequence[:,:-1] = output_sequence
        full_output_sequence[:,-1:] = input_sequence[:,0].reshape(output_sequence.shape[0],1)
        return full_output_sequence[:-self.L2,:], full_output_sequence[-self.L2:,:]
    
    def generateAnomalous(self, anomaly_params, verbose=False):
        xa, xb = self.generate()
        x = np.vstack((xa,xb))
        if np.random.random() < anomaly_params[0]: # If overall anomaly
            anomalies = [False]*len(anomaly_params[1])
            while True not in anomalies:
               for i, probability in enumerate(anomaly_params[1]):
                 anomalies[i] = (np.random.random() < probability)
            
            ranges = []
            min_ts = round((anomaly_params[2][1]-anomaly_params[2][0])*anomaly_params[5])
            for i in anomaly_params[1]:
                start = np.random.randint(anomaly_params[2][0], anomaly_params[2][1]-min_ts)
                stop = np.random.randint(start+min_ts, anomaly_params[2][1])
                ranges.append([start, stop])
                
            x_anomalous = np.copy(x)
            y_anomalous = np.zeros(x.shape)
            for signal, _ in enumerate(anomaly_params[1]):
                if(anomalies[signal]):
                    # Two types of anomalies:
                    # Constant: constant increase/decrease with smoothing
                    # Random: random additions or subtraction per timestep
                    # Two lengths:
                    # Lasting: anomaly stays for the entire remaining sequence
                    # Fleeting: signal returns to normal values at some point
                    a_length = np.random.choice(['l','f'], p=anomaly_params[4][0])
                    a_type = np.random.choice(['c','r'], p=anomaly_params[4][1])
                    op = np.random.choice(['+','-'])
                    
                    if a_length == 'l':
                        length = anomaly_params[2][1] - ranges[signal][0]
                        smoothing = np.hstack((np.linspace(anomaly_params[6], 1, round(length/3)),
                                               np.ones(length-round(length/3)*1)))
                        if a_type == 'r':
                            for idx in range(ranges[signal][0],anomaly_params[2][1]):
                                x_anomalous[idx,signal] = eval("x[idx,signal]"+op+\
                                           "np.random.uniform(anomaly_params[3][0], anomaly_params[3][1])\
                                               *smoothing[idx-ranges[signal][0]]*x[idx,signal]")
                                if((x[idx,signal]-x_anomalous[idx,signal])**2 > 0):
                                    y_anomalous[idx,signal] = 1
                        else:
                            mean_xrange = np.mean(x[ranges[signal][0]:anomaly_params[2][1],signal])    
                            constant_val = np.random.uniform(anomaly_params[3][0], anomaly_params[3][1])*mean_xrange
                            smoothing *= constant_val
                            for idx in range(ranges[signal][0],anomaly_params[2][1]):
                                x_anomalous[idx,signal] = eval("x[idx,signal]"+op+"smoothing[idx-ranges[signal][0]]")
                                if((x[idx,signal]-x_anomalous[idx,signal])**2 > 0):
                                    y_anomalous[idx,signal] = 1
                    else:
                        length = ranges[signal][1] - ranges[signal][0]
                        smoothing = np.hstack((np.linspace(anomaly_params[6], 1, round(length/3)),
                                               np.ones(length-round(length/3)*2),
                                               np.linspace(1, anomaly_params[6], round(length/3))))
                        if a_type == 'r':
                            for idx in range(ranges[signal][0],ranges[signal][1]):
                                x_anomalous[idx,signal] = eval("x[idx,signal]"+op+\
                                           "np.random.uniform(anomaly_params[3][0], anomaly_params[3][1])\
                                               *smoothing[idx-ranges[signal][0]]*x[idx,signal]")
                                if((x[idx,signal]-x_anomalous[idx,signal])**2 > 0):
                                    y_anomalous[idx,signal] = 1
                        else:
                            mean_xrange = np.mean(x[ranges[signal][0]:ranges[signal][1],signal])
                            constant_val = np.random.uniform(anomaly_params[3][0], anomaly_params[3][1])*mean_xrange
                            smoothing *= constant_val
                            for idx in range(ranges[signal][0],ranges[signal][1]):
                                x_anomalous[idx,signal] = eval("x[idx,signal]"+op+"smoothing[idx-ranges[signal][0]]")
                                if((x[idx,signal]-x_anomalous[idx,signal])**2 > 0):
                                    y_anomalous[idx,signal] = 1
                     
            if verbose:
                f, (ax1, ax2, ax3) = plt.subplots(3, figsize=(9,18));
                ax1.plot(x[:,:]);
                ax1.set_title('Original Signal');
                ax2.plot(x_anomalous[:,:]);
                ax2.set_title('Signal + Anomaly');
                diff = x_anomalous[:,:]-x[:,:]
                ax3.plot(diff);
                ax3.set_title('Difference Due to Anomaly');
            
            return x_anomalous, y_anomalous
        else:
            return x, np.zeros(x.shape)
