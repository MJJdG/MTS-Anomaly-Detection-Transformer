#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import numpy as np
import os 
import sklearn
from sklearn.metrics import f1_score

os.environ['TF_KERAS'] = '1'

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.utils import plot_model, HDF5Matrix
from tensorflow.keras.callbacks import EarlyStopping

from keras_radam import RAdam

def mtsLSTM(modelID, fn, L1, L2, sign=21, noLayers=4, noUnits=512, batch_size=128, epochs=300):
    K.clear_session()
    
    print('Loading Data')
    folder = ''
    
    print("Signals:",np.array(HDF5Matrix(folder+fn+'.hdf5', 'column_names')))
    
    #Scale the features using min-max scaler in range -1 -> 1 for tanh activation
    sc = Scaler3DMM(np.array(HDF5Matrix(folder+fn+'.hdf5', 'x_train')))
    x_train = sc.transform(np.array(HDF5Matrix(folder+fn+'.hdf5', 'x_train')))
    y_train = sc.transform(np.array(HDF5Matrix(folder+fn+'.hdf5', 'y_train')))
    x_val = sc.transform(np.array(HDF5Matrix(folder+fn+'.hdf5', 'x_val')))
    y_val = sc.transform(np.array(HDF5Matrix(folder+fn+'.hdf5', 'y_val')))
    
    # Define encoder
    inputEnc = Input(batch_shape=(batch_size, x_train.shape[1], x_train.shape[2]), name='enc_input')
    encoder = LSTM(noUnits, activation='tanh', return_sequences=True, stateful=False, dropout=0.1)(inputEnc)
    if noLayers > 4:
        for extra_layer in range((noLayers-4)//2):
            encoder = LSTM(noUnits, activation='tanh', return_sequences=True, stateful=False, dropout=0.1)(encoder)
    encoder = LSTM(noUnits, activation='tanh', return_sequences=False, stateful=False, dropout=0.1)(encoder)
    
    # Define bottleneck
    bottleneck = RepeatVector(L2)(encoder)
    
    # Define reconstruction decoder
    decoder = LSTM(noUnits, activation='tanh', return_sequences=True, stateful=False, dropout=0.1)(bottleneck)
    if noLayers > 4:
        for extra_layer in range((noLayers-4)//2):
            decoder = LSTM(noUnits, activation='tanh', return_sequences=True, stateful=False, dropout=0.1)(decoder)
    decoder = LSTM(noUnits, activation='tanh', return_sequences=True, stateful=False, dropout=0.1)(decoder)
    decoder = TimeDistributed(Dense(sign, activation='linear', name='dec_output'))(decoder)
    
    # Create model and compile
    opt = RAdam(learning_rate=1e-4, total_steps=10000, warmup_proportion=0.1, min_lr=1e-7)
    model = Model(inputs=inputEnc, outputs=decoder)
    model.compile(loss=root_mean_squared_error, 
                  optimizer=opt,
                  metrics=[metricWrapper(m) for m in range(sign)])
    plot_model(model, show_shapes=True, to_file='LSTM_AE'+modelID+'.png')
    model.summary()
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=.00001, patience=3, restore_best_weights=True)
    
    # fit model
    print('Training Model...')
    history = model.fit(x_train, 
                        y_train,
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(x_val, y_val),
                        verbose=1,
                        callbacks=[es],
                        shuffle=True)
    
    # Reduce memory load
    del x_train, y_train, x_val, y_val
    
    x_test = sc.transform(np.array(HDF5Matrix(folder+fn+'.hdf5', 'x_test')))
    y_test = sc.transform(np.array(HDF5Matrix(folder+fn+'.hdf5', 'y_test')))
    
    # Evaluate the model on the test data
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('test metrics:', results)
    
    x_anomalous = sc.transform(np.array(HDF5Matrix(folder+fn+'.hdf5', 'x_anomalous')))
    y_anomalous = np.array(HDF5Matrix(folder+fn+'.hdf5', 'y_anomalous'))
    
    x_anompred = np.zeros(x_anomalous.shape)
    x_anompred[:,:L1,:] = x_anomalous[:,:L1,:]
    
    for batch in range(x_anompred.shape[0]//batch_size):
        print("Batch:",batch)
        x_anompred[batch*batch_size:(batch+1)*batch_size,L1:,:] = model.predict(x_anompred[batch*batch_size:(batch+1)*batch_size,:L1,:], batch_size=batch_size)
    
    labels = np.amax(y_anomalous[:,L1:,:sign-1], axis=1)
    max_rse_prediction = np.sqrt(((x_anomalous[:,L1:,:sign-1] - x_anompred[:,L1:,:sign-1]) ** 2)).max(axis=1)
    predictions = np.zeros(max_rse_prediction.shape)
    for idx, sample in enumerate(max_rse_prediction):
        predictions[idx] = (sample >= [x for x in results[1:sign]])
    
    print("Signal anomaly prediction:")
    meanF1 = 0
    for i in range(sign-1):
        meanF1 += sklearn.metrics.f1_score(labels[:,i], predictions[:,i], average='weighted')
        print("F1 signal",i,sklearn.metrics.f1_score(labels[:,i], predictions[:,i], average='weighted'))
    meanF1a = meanF1/(sign-1)
    print("Mean F1 signal", meanF1a)
    
    labels = y_anomalous[:,L1:,:sign-1]
    max_rse_prediction = np.sqrt(((x_anomalous[:,L1:,:sign-1] - x_anompred[:,L1:,:sign-1]) ** 2))
    predictions = np.zeros(max_rse_prediction.shape)
    for idx, sample in enumerate(max_rse_prediction):
        for jdx, step in enumerate(sample):
            predictions[idx,jdx,:] = (step >= [x for x in results[1:sign]])
    
    print("Signal x Ts anomaly prediction:")
    meanF1 = 0
    for i in range(sign-1):
        meanF1 += sklearn.metrics.f1_score(labels[:,:,i], predictions[:,:,i], average='weighted')
        print("F1 signal",i,sklearn.metrics.f1_score(labels[:,:,i], predictions[:,:,i], average='weighted'))
    meanF1b = meanF1/(sign-1)
    print("Mean F1 signal", meanF1b)

    # Write results to file
    with open("LSTM_results.txt", "a") as file:
        file.write('test metrics model %s:\n' % modelID)
        for item in results:
            file.write("%s\n" % item)
        file.write('meanF1a: %s\n' % meanF1a)
        file.write('meanF1b: %s\n' % meanF1b)

class Scaler3DMM():

    def __init__(self,X,y=None):
        self.maxs, self.mins = [], []
        for signal in X.T:
            self.maxs.append(np.amax(signal))
            self.mins.append(np.amin(signal))
        self.scalef = max(self.maxs)/(self.maxs)
        print(self.scalef)
            
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

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))

def metricWrapper(m):
    def max_rse(y_true, y_pred):
        return K.sqrt(K.max(K.square(y_true[m,:] - y_pred[m,:]), axis=0))
    max_rse.__name__ = 'max_rse' + str(m)
    return max_rse

if __name__ == '__main__':
    ### Seed 1
    mtsLSTM('m1', 'synthetic_L1_300_L2_30_SIG_20_SEED_77', 300, 30, 21, 4, 256)
    mtsLSTM('m2', 'synthetic_L1_300_L2_30_SIG_20_SEED_77', 300, 30, 21, 4, 512)
    mtsLSTM('m3', 'synthetic_L1_300_L2_30_SIG_20_SEED_77', 300, 30, 21, 4, 1024)
    mtsLSTM('m4', 'synthetic_L1_300_L2_30_SIG_20_SEED_77', 300, 30, 21, 6, 256)
    mtsLSTM('m5', 'synthetic_L1_300_L2_30_SIG_20_SEED_77', 300, 30, 21, 6, 512)
    mtsLSTM('m6', 'synthetic_L1_300_L2_30_SIG_20_SEED_77', 300, 30, 21, 6, 1024)
    
    mtsLSTM('m7', 'synthetic_L1_300_L2_30_SIG_40_SEED_77', 300, 30, 41, 4, 256)
    mtsLSTM('m8', 'synthetic_L1_300_L2_30_SIG_40_SEED_77', 300, 30, 41, 4, 512)
    mtsLSTM('m9', 'synthetic_L1_300_L2_30_SIG_40_SEED_77', 300, 30, 41, 4, 1024)
    mtsLSTM('m10', 'synthetic_L1_300_L2_30_SIG_40_SEED_77', 300, 30, 41, 6, 256)
    mtsLSTM('m11', 'synthetic_L1_300_L2_30_SIG_40_SEED_77', 300, 30, 41, 6, 512)
    mtsLSTM('m12', 'synthetic_L1_300_L2_30_SIG_40_SEED_77', 300, 30, 41, 6, 1024)
    
    mtsLSTM('m13', 'synthetic_L1_600_L2_60_SIG_20_SEED_77', 600, 60, 21, 4, 256)
    mtsLSTM('m14', 'synthetic_L1_600_L2_60_SIG_20_SEED_77', 600, 60, 21, 4, 512)
    mtsLSTM('m15', 'synthetic_L1_600_L2_60_SIG_20_SEED_77', 600, 60, 21, 4, 1024)
    mtsLSTM('m16', 'synthetic_L1_600_L2_60_SIG_20_SEED_77', 600, 60, 21, 6, 256)
    mtsLSTM('m17', 'synthetic_L1_600_L2_60_SIG_20_SEED_77', 600, 60, 21, 6, 512)
    mtsLSTM('m18', 'synthetic_L1_600_L2_60_SIG_20_SEED_77', 600, 60, 21, 6, 1024)
    
    mtsLSTM('m19', 'synthetic_L1_600_L2_60_SIG_40_SEED_77', 600, 60, 41, 4, 256) 
    mtsLSTM('m20', 'synthetic_L1_600_L2_60_SIG_40_SEED_77', 600, 60, 41, 4, 512)
    mtsLSTM('m21', 'synthetic_L1_600_L2_60_SIG_40_SEED_77', 600, 60, 41, 4, 1024)
    mtsLSTM('m22', 'synthetic_L1_600_L2_60_SIG_40_SEED_77', 600, 60, 41, 6, 256)
    mtsLSTM('m23', 'synthetic_L1_600_L2_60_SIG_40_SEED_77', 600, 60, 41, 6, 512)
    mtsLSTM('m24', 'synthetic_L1_600_L2_60_SIG_40_SEED_77', 600, 60, 41, 6, 1024)
    ### Seed 2
    mtsLSTM('m25', 'synthetic_L1_300_L2_30_SIG_20_SEED_63', 300, 30, 21, 4, 256)
    mtsLSTM('m26', 'synthetic_L1_300_L2_30_SIG_20_SEED_63', 300, 30, 21, 4, 512)
    mtsLSTM('m27', 'synthetic_L1_300_L2_30_SIG_20_SEED_63', 300, 30, 21, 4, 1024)
    mtsLSTM('m28', 'synthetic_L1_300_L2_30_SIG_20_SEED_63', 300, 30, 21, 6, 256)
    mtsLSTM('m29', 'synthetic_L1_300_L2_30_SIG_20_SEED_63', 300, 30, 21, 6, 512)
    mtsLSTM('m30', 'synthetic_L1_300_L2_30_SIG_20_SEED_63', 300, 30, 21, 6, 1024)
    
    mtsLSTM('m31', 'synthetic_L1_300_L2_30_SIG_40_SEED_63', 300, 30, 41, 4, 256)
    mtsLSTM('m32', 'synthetic_L1_300_L2_30_SIG_40_SEED_63', 300, 30, 41, 4, 512)
    mtsLSTM('m33', 'synthetic_L1_300_L2_30_SIG_40_SEED_63', 300, 30, 41, 4, 1024)
    mtsLSTM('m34', 'synthetic_L1_300_L2_30_SIG_40_SEED_63', 300, 30, 41, 6, 256)
    mtsLSTM('m35', 'synthetic_L1_300_L2_30_SIG_40_SEED_63', 300, 30, 41, 6, 512)
    mtsLSTM('m36', 'synthetic_L1_300_L2_30_SIG_40_SEED_63', 300, 30, 41, 6, 1024)
    
    mtsLSTM('m37', 'synthetic_L1_600_L2_60_SIG_20_SEED_63', 600, 60, 21, 4, 256)
    mtsLSTM('m38', 'synthetic_L1_600_L2_60_SIG_20_SEED_63', 600, 60, 21, 4, 512)
    mtsLSTM('m39', 'synthetic_L1_600_L2_60_SIG_20_SEED_63', 600, 60, 21, 4, 1024)
    mtsLSTM('m40', 'synthetic_L1_600_L2_60_SIG_20_SEED_63', 600, 60, 21, 6, 256)
    mtsLSTM('m41', 'synthetic_L1_600_L2_60_SIG_20_SEED_63', 600, 60, 21, 6, 512)
    mtsLSTM('m42', 'synthetic_L1_600_L2_60_SIG_20_SEED_63', 600, 60, 21, 6, 1024)

    mtsLSTM('m43', 'synthetic_L1_600_L2_60_SIG_40_SEED_63', 600, 60, 41, 4, 256) 
    mtsLSTM('m44', 'synthetic_L1_600_L2_60_SIG_40_SEED_63', 600, 60, 41, 4, 512)
    mtsLSTM('m45', 'synthetic_L1_600_L2_60_SIG_40_SEED_63', 600, 60, 41, 4, 1024)
    mtsLSTM('m46', 'synthetic_L1_600_L2_60_SIG_40_SEED_63', 600, 60, 41, 6, 256)
    mtsLSTM('m47', 'synthetic_L1_600_L2_60_SIG_40_SEED_63', 600, 60, 41, 6, 512)
    mtsLSTM('m48', 'synthetic_L1_600_L2_60_SIG_40_SEED_63', 600, 60, 41, 6, 1024)
    ### Seed 3
    mtsLSTM('m49', 'synthetic_L1_300_L2_30_SIG_20_SEED_12', 300, 30, 21, 4, 256)
    mtsLSTM('m50', 'synthetic_L1_300_L2_30_SIG_20_SEED_12', 300, 30, 21, 4, 512)
    mtsLSTM('m51', 'synthetic_L1_300_L2_30_SIG_20_SEED_12', 300, 30, 21, 4, 1024)
    mtsLSTM('m52', 'synthetic_L1_300_L2_30_SIG_20_SEED_12', 300, 30, 21, 6, 256)
    mtsLSTM('m53', 'synthetic_L1_300_L2_30_SIG_20_SEED_12', 300, 30, 21, 6, 512)
    mtsLSTM('m54', 'synthetic_L1_300_L2_30_SIG_20_SEED_12', 300, 30, 21, 6, 1024)
    
    mtsLSTM('m55', 'synthetic_L1_300_L2_30_SIG_40_SEED_12', 300, 30, 41, 4, 256)
    mtsLSTM('m56', 'synthetic_L1_300_L2_30_SIG_40_SEED_12', 300, 30, 41, 4, 512)
    mtsLSTM('m57', 'synthetic_L1_300_L2_30_SIG_40_SEED_12', 300, 30, 41, 4, 1024)
    mtsLSTM('m58', 'synthetic_L1_300_L2_30_SIG_40_SEED_12', 300, 30, 41, 6, 256)
    mtsLSTM('m59', 'synthetic_L1_300_L2_30_SIG_40_SEED_12', 300, 30, 41, 6, 512)
    mtsLSTM('m60', 'synthetic_L1_300_L2_30_SIG_40_SEED_12', 300, 30, 41, 6, 1024)
    
    mtsLSTM('m61', 'synthetic_L1_600_L2_60_SIG_20_SEED_12', 600, 60, 21, 4, 256)
    mtsLSTM('m62', 'synthetic_L1_600_L2_60_SIG_20_SEED_12', 600, 60, 21, 4, 512)
    mtsLSTM('m63', 'synthetic_L1_600_L2_60_SIG_20_SEED_12', 600, 60, 21, 4, 1024)
    mtsLSTM('m64', 'synthetic_L1_600_L2_60_SIG_20_SEED_12', 600, 60, 21, 6, 256)
    mtsLSTM('m65', 'synthetic_L1_600_L2_60_SIG_20_SEED_12', 600, 60, 21, 6, 512)
    mtsLSTM('m66', 'synthetic_L1_600_L2_60_SIG_20_SEED_12', 600, 60, 21, 6, 1024)

    mtsLSTM('m67', 'synthetic_L1_600_L2_60_SIG_40_SEED_12', 600, 60, 41, 4, 256) 
    mtsLSTM('m68', 'synthetic_L1_600_L2_60_SIG_40_SEED_12', 600, 60, 41, 4, 512)
    mtsLSTM('m69', 'synthetic_L1_600_L2_60_SIG_40_SEED_12', 600, 60, 41, 4, 1024)
    mtsLSTM('m70', 'synthetic_L1_600_L2_60_SIG_40_SEED_12', 600, 60, 41, 6, 256)
    mtsLSTM('m71', 'synthetic_L1_600_L2_60_SIG_40_SEED_12', 600, 60, 41, 6, 512)
    mtsLSTM('m72', 'synthetic_L1_600_L2_60_SIG_40_SEED_12', 600, 60, 41, 6, 1024)