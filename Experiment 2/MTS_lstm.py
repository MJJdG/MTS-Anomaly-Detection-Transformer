#!/usr/bin/env python
# coding: utf-8
"""
Implementation of a LSTM for multivariate time series forecasting and anomaly detection
@author: Max de Grauw (M.degrauw@student.ru.nl)
"""

from __future__ import print_function
import numpy as np
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
import os 

os.environ['TF_KERAS'] = '1'

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.utils import plot_model, HDF5Matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import mean_squared_error

from keras_radam import RAdam

from MTS_utils import Scaler3DMM_tanh

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))

def metricWrapper(m):
    def max_rse(y_true, y_pred):
        return K.sqrt(K.max(K.square(y_true[m,:] - y_pred[m,:]), axis=0))
    max_rse.__name__ = 'max_rse' + str(m)
    return max_rse

def mtsLSTM(modelID, fn, L1, L2, sign=21, noLayers=4, noUnits=512, batch_size=64, epochs=300):
    K.clear_session()
    
    print('Loading Data')
    folder = ''
    
    print("Signals:",np.array(HDF5Matrix(folder+fn+'.hdf5', 'column_names')))
    
    #Scale the features using min-max scaler in range -1 -> 1 for tanh activation
    sc = Scaler3DMM_tanh(np.array(HDF5Matrix(folder+fn+'.hdf5', 'x_train')))
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
    meanF1, mean_accSa, mean_hamLa = 0, 0, 0
    for i in range(sign-1):
        meanF1 += f1_score(labels[:,i], predictions[:,i], average='weighted')
        mean_accSa += accuracy_score(labels[:,i], predictions[:,i])
        mean_hamLa += hamming_loss(labels[:,i], predictions[:,i])
        #print("F1 signal", i, f1_score(labels[:,i], predictions[:,i], average='weighted'))
    
    meanF1a = meanF1/(sign-1)
    mean_accSa = mean_accSa/(sign-1)
    mean_hamLa = mean_hamLa/(sign-1)
    
    print("Mean F1 signal", meanF1a)
    print("accuracy_score:", mean_accSa)
    print("Hamming_loss:", mean_hamLa)
    
    labels = y_anomalous[:,L1:,:sign-1]
    max_rse_prediction = np.sqrt(((x_anomalous[:,L1:,:sign-1] - x_anompred[:,L1:,:sign-1]) ** 2))
    predictions = np.zeros(max_rse_prediction.shape)
    for idx, sample in enumerate(max_rse_prediction):
        for jdx, step in enumerate(sample):
            predictions[idx,jdx,:] = (step >= [x for x in results[1:sign]])
    
    print("Signal x Ts anomaly prediction:")
    meanF1 = 0
    mean_accSb, mean_hamLb = np.zeros(sign-1), np.zeros(sign-1)
    for i in range(sign-1):
        meanF1 += f1_score(labels[:,:,i], predictions[:,:,i], average='weighted')
        #print("F1 signal", i, f1_score(labels[:,:,i], predictions[:,:,i], average='weighted'))
        for j in range(labels.shape[0]):
            mean_accSb[i] += accuracy_score(labels[j,:,i], predictions[j,:,i])
            mean_hamLb[i] += hamming_loss(labels[j,:,i], predictions[j,:,i])
            
    meanF1b = meanF1/(sign-1)
    mean_accSb = np.mean(mean_accSb/labels.shape[0])
    mean_hamLb = np.mean(mean_hamLb/labels.shape[0])
    
    print("Mean F1 signal", meanF1b)
    print("accuracy_score:", mean_accSb)
    print("Hamming_loss:", mean_hamLb)
    
    # Write results to file
    with open("LSTM_results.txt", "a") as file:
        file.write('test metrics model %s:\n' % modelID)
        file.write("test RMSE: %s\n" % results[0])
        for item in results[1:]:
            file.write("%s\n" % item)
        file.write('meanF1a: %s\n' % meanF1a)
        file.write("accuracy_score_a: %s\n" % mean_accSa)
        file.write("Hamming_loss_a: %s\n" % mean_hamLa)
        file.write('meanF1b: %s\n' % meanF1b)
        file.write("accuracy_score_b: %s\n" % mean_accSb)
        file.write("Hamming_loss_b: %s\n" % mean_hamLb)
        
    return history
