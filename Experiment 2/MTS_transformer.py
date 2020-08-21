#!/usr/bin/env python
# coding: utf-8
"""
Implementation of a Transformer for multivariate time series forecasting and anomaly detection
@author: Max de Grauw (M.degrauw@student.ru.nl)
BUilding upon the base Transformer implementation of github user 'huseinzol05'
https://github.com/huseinzol05/Stock-Prediction-Models/blob/master/deep-learning/16.attention-is-all-you-need.ipynb
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, hamming_loss, accuracy_score 
from sklearn.model_selection import KFold
import gc

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

from tensorflow.keras.utils import HDF5Matrix
from keras_radam.training import RAdamOptimizer

from MTS_utils import Scaler3DMM

def layer_norm(inputs, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))

    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())
    
    outputs = gamma * normalized + beta
    return outputs

def multihead_attn(queries, keys, q_masks, k_masks, future_binding, num_units, num_heads):
    
    T_q = tf.shape(queries)[1]                                      
    T_k = tf.shape(keys)[1]                  

    Q = tf.layers.dense(queries, num_units, name='Q')                              
    K_V = tf.layers.dense(keys, 2*num_units, name='K_V')    
    K, V = tf.split(K_V, 2, -1)        

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)                         
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)                    
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)                      

    align = tf.matmul(Q_, tf.transpose(K_, [0,2,1]))                      
    align = align / np.sqrt(K_.get_shape().as_list()[-1])                 

    paddings = tf.fill(tf.shape(align), float('-inf'))                   

    key_masks = k_masks                                                 
    key_masks = tf.tile(key_masks, [num_heads, 1])                       
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, T_q, 1])            
    align = tf.where(tf.equal(key_masks, 0), paddings, align)       

    if future_binding:
        lower_tri = tf.ones([T_q, T_k])                                          
        lower_tri = tf.linalg.LinearOperatorLowerTriangular(lower_tri).to_dense()  
        masks = tf.tile(tf.expand_dims(lower_tri,0), [tf.shape(align)[0], 1, 1]) 
        align = tf.where(tf.equal(masks, 0), paddings, align)                      
    
    align = tf.nn.softmax(align)                                            
    query_masks = tf.to_float(q_masks)                                             
    query_masks = tf.tile(query_masks, [num_heads, 1])                             
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, T_k])            
    align *= query_masks
    
    outputs = tf.matmul(align, V_)                                                 
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)             
    outputs += queries                                                             
    outputs = layer_norm(outputs)                                                 
    return outputs

def pointwise_feedforward(inputs, hidden_units, activation=None):
    outputs = tf.layers.dense(inputs, 4*hidden_units, activation=activation)
    outputs = tf.layers.dense(outputs, hidden_units, activation=None)
    outputs += inputs
    outputs = layer_norm(outputs)
    return outputs

def sinusoidal_position_encoding(inputs, mask, repr_dim):
    T = tf.shape(inputs)[1]
    pos = tf.reshape(tf.range(0.0, tf.to_float(T), dtype=tf.float32), [-1, 1])
    i = np.arange(0, repr_dim, 2, np.float32)
    denom = np.reshape(np.power(10000.0, i / repr_dim), [1, -1])
    enc = tf.expand_dims(tf.concat([tf.sin(pos / denom), tf.cos(pos / denom)], 1), 0)
    return tf.tile(enc, [tf.shape(inputs)[0], 1, 1]) * tf.expand_dims(tf.to_float(mask), -1)

def label_smoothing(inputs, epsilon=0.1):
    C = inputs.get_shape().as_list()[-1]
    return ((1 - epsilon) * inputs) + (epsilon / C)

def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer
    
def mtsTransformerForecasting(modelID, fn, L1, L2, sign=21, num_blocks=1, size_embedding=512, num_heads=16,
                              batch_size=32, epochs=300, learning_rate =1e-4):
    
    ######################################################################################
    # Read data
    ######################################################################################
    print('Loading Data...')
    folder = ''
    
    print("Signals:",np.array(HDF5Matrix(folder+fn+'.hdf5', 'column_names')))
    
    #Scale the features using min-max scaler in range 0 -> 1
    sc = Scaler3DMM(np.array(HDF5Matrix(folder+fn+'.hdf5', 'x_train')))
    
    x_train = sc.transform(np.array(HDF5Matrix(folder+fn+'.hdf5', 'x_train')))
    y_train = sc.transform(np.array(HDF5Matrix(folder+fn+'.hdf5', 'y_train')))
    x_val = sc.transform(np.array(HDF5Matrix(folder+fn+'.hdf5', 'x_val')))
    y_val = sc.transform(np.array(HDF5Matrix(folder+fn+'.hdf5', 'y_val')))
    x_test = sc.transform(np.array(HDF5Matrix(folder+fn+'.hdf5', 'x_test')))
    y_test = sc.transform(np.array(HDF5Matrix(folder+fn+'.hdf5', 'y_test')))
    x_anomalous = sc.transform(np.array(HDF5Matrix(folder+fn+'.hdf5', 'x_anomalous')))
    y_anomalous = np.array(HDF5Matrix(folder+fn+'.hdf5', 'y_anomalous'))

    ######################################################################################
    # Initialize model
    ######################################################################################
    tf.reset_default_graph()

    forecasting_model = Forecasting_Attention(size_embedding, learning_rate, x_train.shape[1],\
                                              y_train.shape[2], num_blocks, num_heads, L2)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    ######################################################################################
    # Training loop forecasting
    ######################################################################################
    pbar = tqdm(range(epochs), desc = 'train loop')
    prev_rmse, counter = 1e3, 0
    for i in pbar:
        total_rmse = []
        max_rse = np.zeros((L2, sign))
        for k in range(batch_size, x_train.shape[0], batch_size):
            logits, _, rmse, indiv_max_rse = sess.run(
                [forecasting_model.logits, forecasting_model.optimizer,\
                 forecasting_model.rmse, forecasting_model.indiv_max_rse],
                feed_dict = {
                    forecasting_model.X: x_train[k-batch_size:k, :, :],
                    forecasting_model.Y: y_train[k-batch_size:k]
                },
            ) 
            total_rmse.append(rmse)
            max_rse += indiv_max_rse
        pbar.set_postfix(rmse = np.mean(total_rmse), max_rse = (max_rse/(x_train.shape[0]/batch_size))[0])

        ######################################################################################
        # Evaluate the model on the validation data
        ######################################################################################
        val_rmse = 0
        val_indiv_max_rse = np.zeros((x_val.shape[0], L2, sign))
        for l in range(batch_size, x_val.shape[0], batch_size):
            out_logits = sess.run(
                forecasting_model.logits,
                feed_dict = {
                    forecasting_model.X: x_val[l-batch_size:l, :, :],
                },
            ) 
            val_rmse += np.sqrt(np.mean(np.square(y_val[l-batch_size:l] - out_logits)))
            val_indiv_max_rse[l-batch_size:l] = np.sqrt(np.max(np.square(y_val[l-batch_size:l] - out_logits),axis=0))
        print("val_rmse:",val_rmse/(x_val.shape[0]/batch_size))
        print("val_max_rse:", np.max(np.max(val_indiv_max_rse, axis=0), axis=0))    
        if prev_rmse-val_rmse/(L2*(x_val.shape[0]/batch_size)) <= .00001: # Early stopping
            counter += 1
        if counter == 3:
            print("Early stopping at epoch",i)
            break
        prev_rmse = val_rmse/(L2*(x_val.shape[0]/batch_size))

    ######################################################################################        
    # Evaluate the model on the test data
    ######################################################################################
    pbar = tqdm(range(1), desc = 'test loop')
    for i in pbar:
        test_rmse = 0
        test_indiv_max_rse = np.zeros((x_test.shape[0], L2, sign))
        for k in range(batch_size, x_test.shape[0], batch_size):
            out_logits = sess.run(
                forecasting_model.logits,
                feed_dict = {
                    forecasting_model.X: x_test[k-batch_size:k, :, :],
                },
            ) 
            test_rmse += np.sqrt(np.mean(np.square(y_test[k-batch_size:k] - out_logits)))
            test_indiv_max_rse[k-batch_size:k] = np.sqrt(np.max(np.square(y_test[k-batch_size:k] - out_logits),axis=0))
        pbar.set_postfix(test_rmse = test_rmse/(x_test.shape[0]/batch_size),\
                         test_max_rse = np.max(np.max(test_indiv_max_rse, axis=0), axis=0))
        thresholds = np.max(test_indiv_max_rse, axis=0)

    ######################################################################################
    # Predict future ts using last n days as a start for anomaly detection
    ######################################################################################
    x_anompred = np.zeros(x_anomalous.shape)
    x_anompred_ffw = np.zeros((x_anomalous.shape[0], L1, size_embedding))
    pbar = tqdm(range(batch_size, x_anomalous.shape[0], batch_size), desc = "anom pred")
    anom_rmse = 0
    anom_indiv_max_rse = np.zeros(x_anomalous.shape)
    for k in pbar:
        batch_pred = np.zeros((batch_size, x_anomalous.shape[1], x_anomalous.shape[2]))
        batch_pred[:,:L1,:] = x_anomalous[k-batch_size:k,:L1,:]
        out_logits, out_ffw = sess.run(
            [forecasting_model.logits, forecasting_model.feedforward_outputs],
            feed_dict = {
                forecasting_model.X:batch_pred[:,:L1,:]
            },
        )
        anom_rmse += np.sqrt(np.mean(np.square(x_anomalous[k-batch_size:k,L1:] - out_logits)))
        anom_indiv_max_rse[k-batch_size:k,L1:] = np.sqrt(np.max(np.square(x_anomalous[k-batch_size:k,L1:] - out_logits),axis=0))
        batch_pred[:, L1:, :] = out_logits
        x_anompred[k-batch_size:k, :, :] = batch_pred
        x_anompred_ffw[k-batch_size:k, :, :] = out_ffw

    print("Mean RMSE anomaly forecasting:", anom_rmse/(x_anomalous.shape[0]/batch_size))
    print("Individual Max RSE anomaly forecasting:", np.max(np.max(anom_indiv_max_rse, axis=0), axis=0))

    print("Signal anomaly prediction:")
    labels = np.amax(y_anomalous[:,L1:,:sign-1], axis=1)
    max_rse_prediction = np.sqrt(np.max(np.square(x_anomalous[:,L1:,:sign-1] - x_anompred[:,L1:,:sign-1]),axis=1))
    predictions = np.zeros(max_rse_prediction.shape)
    for idx, sample in enumerate(max_rse_prediction):
        predictions[idx] = (sample >= [x for x in np.max(thresholds, axis=0)[:sign-1]])

    meanF1, mean_accSa, mean_hamLa = 0, 0, 0
    for i in range(sign-1):
        meanF1 += f1_score(labels[:,i], predictions[:,i], average='weighted')
        mean_accSa += accuracy_score(labels[:,i], predictions[:,i])
        mean_hamLa += hamming_loss(labels[:,i], predictions[:,i])
        #print("F1 signal", i, f1_score(labels[:,i], predictions[:,i], average='weighted'))
    
    meanF1a = meanF1/(sign-1)
    mean_accSa = mean_accSa/(sign-1)
    mean_hamLa = mean_hamLa/(sign-1)
    
    print("Mean F1 signal",meanF1a)
    print("accuracy_score:", mean_accSa)
    print("Hamming_loss:", mean_hamLa)
    
    print("Signal x Ts anomaly prediction:")
    labels = y_anomalous[:,L1:,:sign-1]
    max_rse_prediction = np.sqrt(np.square(x_anomalous[:,L1:,:sign-1] - x_anompred[:,L1:,:sign-1]))
    predictions = np.zeros(max_rse_prediction.shape)
    for idx, sample in enumerate(max_rse_prediction):
        for jdx, step in enumerate(sample):
            predictions[idx,jdx,:] = (step >= [x for x in thresholds[jdx,:sign-1]])

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

    print("Mean F1 signal",meanF1b)
    print("accuracy_score:", mean_accSb)
    print("Hamming_loss:", mean_hamLb)
    
    ######################################################################################
    # Write results to file
    ######################################################################################
    with open("Transformer_results.txt", "a") as file:
        file.write('test metrics model %s:\n' % modelID)
        file.write("test RMSE: %s\n" % (test_rmse/(x_test.shape[0]/batch_size)))
        for item in np.max(np.max(test_indiv_max_rse, axis=0), axis=0):
            file.write("%s\n" % item)
        file.write('meanF1a: %s\n' % meanF1a)
        file.write("accuracy_score_a: %s\n" % mean_accSa)
        file.write("Hamming_loss_b: %s\n" % mean_hamLa)
        file.write('meanF1b: %s\n' % meanF1b)
        file.write("accuracy_score_b: %s\n" % mean_accSb)
        file.write("Hamming_loss_b: %s\n" % mean_hamLb)
        
    ######################################################################################        
    #Plot first anomalous samples
    ######################################################################################
    for i in range(4):
        f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(16,10));
        ax1.plot(x_anomalous[i,L1:]);
        ax1.set_title('True Signal');
        ax2.plot(x_anompred[i,L1:]);
        ax2.set_title('Predicted Signal');

        difference = np.sqrt(np.square(x_anomalous[i,L1:]-x_anompred[i,L1:]))
        ax3.plot(difference);
        ax3.set_title('Absolute Difference');
        ax4.plot(y_anomalous[i][L1:,:]);
        ax4.set_title('Anomalies');
        
    sess.close()    
    
    del forecasting_model, sc, x_train, y_train, x_val, y_val, x_test, y_test, x_anomalous, y_anomalous   
    gc.collect()
        
    return x_anompred, x_anompred_ffw

def mtsTransformerPrediction(modelID, prediction, encoder_output, fn, L1, L2, sign=21, num_blocks=1, 
                             size_embedding=512, num_heads=16, batch_size=32, epochs=300,
                             learning_rate =1e-4, loss_weight=False, folds=4):
    
    ######################################################################################
    # Read data
    ######################################################################################
    print('Loading Data...')
    folder = ''
    
    #Scale the features using min-max scaler in range 0 -> 1
    sc = Scaler3DMM(np.array(HDF5Matrix(folder+fn+'.hdf5', 'x_train')))
    
    x_anomalous = sc.transform(np.array(HDF5Matrix(folder+fn+'.hdf5', 'x_anomalous')))
    y_anomalous = np.array(HDF5Matrix(folder+fn+'.hdf5', 'y_anomalous'))
    
    ######################################################################################
    # K-fold cross validation
    ######################################################################################
    
    kf = KFold(n_splits=folds, shuffle=False, random_state=None)
    fold = 1
    fold_meanLoss, fold_meanF1a, fold_meanF1b = 0, 0, 0 
    fold_mean_accSa, fold_mean_hamLa, fold_mean_accSb, fold_mean_hamLb = 0, 0, 0, 0
    for train_index, test_index in kf.split(x_anomalous, y_anomalous):
        print("FOLD: ",fold)
        encoder_train, encoder_test = encoder_output[train_index], encoder_output[test_index]
        prediction_train, prediction_test = prediction[train_index], prediction[test_index]
        X_train, X_test = x_anomalous[train_index], x_anomalous[test_index]
        y_train, y_test = y_anomalous[train_index], y_anomalous[test_index]
    
        ######################################################################################
        # Intialize model
        ######################################################################################
        tf.reset_default_graph()

        if loss_weight:
            _, counts = np.unique(y_anomalous, return_counts=True)
            if not isinstance(loss_weight, int): loss_weight = (counts[0]/counts[1])
        else:
            loss_weight = 1

        prediction_model = Prediction_Attention(L1, sign, size_embedding, learning_rate,\
                                                num_blocks, num_heads, batch_size, L2, loss_weight)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        ######################################################################################
        # Training loop
        ######################################################################################
        pbar = tqdm(range(epochs), desc = 'anomaly train loop')
        for i in pbar:
            total_loss = []
            for k in range(batch_size, X_train.shape[0], batch_size):
                _, loss = sess.run(
                    [prediction_model.optimizer, prediction_model.loss],
                    feed_dict = {
                        prediction_model.EncoderOutput: encoder_train[k-batch_size:k],
                        prediction_model.Xtrue: prediction_train[k-batch_size:k, -L2:, :],
                        prediction_model.Xanom: X_train[k-batch_size:k, -L2:, :],
                        prediction_model.Yanom: y_train[k-batch_size:k, -L2:, :]
                    },
                ) 
                total_loss.append(loss)
            pbar.set_postfix(loss = np.mean(total_loss))

        ######################################################################################
        # Testing loop
        ######################################################################################
        pred = np.zeros((X_test.shape[0], L2, sign-1))
        for k in range(batch_size, X_test.shape[0], batch_size):
            logits, los = sess.run(
                [prediction_model.logits, prediction_model.loss],
                feed_dict = {
                    prediction_model.EncoderOutput: encoder_test[k-batch_size:k],
                    prediction_model.Xtrue: prediction_test[k-batch_size:k, -L2:, :],
                    prediction_model.Xanom: X_test[k-batch_size:k, -L2:, :],
                    prediction_model.Yanom: y_test[k-batch_size:k, -L2:, :]
                },
            ) 
            total_loss.append(loss)
            pred[k-batch_size:k,:,:] = logits[:,:,:sign-1]
        print("Test loss:", np.mean(total_loss))
        fold_meanLoss += np.mean(total_loss)     
        
        ######################################################################################
        # Predict and calculate F1-score
        ######################################################################################
        print("Signal anomaly prediction:")
        labels = np.amax(y_test[:,-L2:,:sign-1], axis=1)
        predictions = np.zeros((pred.shape[0], pred.shape[2]))
        for idx, sample in enumerate(np.amax(pred, axis=1)):
            predictions[idx] = np.max(sample) >= 0.5
        
        meanF1, mean_accSa, mean_hamLa = 0, 0, 0
        for i in range(sign-1):
            meanF1 += f1_score(labels[:,i], predictions[:,i], average='weighted')
            mean_accSa += accuracy_score(labels[:,i], predictions[:,i])
            mean_hamLa += hamming_loss(labels[:,i], predictions[:,i])
            #print("F1 signal", i, f1_score(labels[:,i], predictions[:,i], average='weighted'))
                
        meanF1a = meanF1/(sign-1)
        mean_accSa = mean_accSa/(sign-1)
        mean_hamLa = mean_hamLa/(sign-1)
        
        print("Mean F1 signal",meanF1a)
        print("accuracy_score:", mean_accSa)
        print("Hamming_loss:", mean_hamLa)
        
        fold_meanF1a += meanF1a
        fold_mean_accSa += mean_accSa
        fold_mean_hamLa += mean_hamLa
        
        print("Signal x Ts anomaly prediction:")
        labels = y_test[:,-L2:,:sign-1]
        predictions = pred >= 0.5
        
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
        
        print("Mean F1 signal",meanF1b)
        print("accuracy_score:", mean_accSb)
        print("Hamming_loss:", mean_hamLb)
        
        fold_meanF1b += meanF1b
        fold_mean_accSb += mean_accSb
        fold_mean_hamLb += mean_hamLb
    
        ######################################################################################
        # Write results to file
        ######################################################################################
        with open("Transformer_results.txt", "a") as file:
            file.write("Supervised, fold %s:\n" % fold)
            file.write("Test loss: %s\n" % np.mean(total_loss))
            file.write('meanF1a: %s\n' % meanF1a)
            file.write('meanF1b: %s\n' % meanF1b)
            file.write('accSa: %s\n' % mean_accSa)
            file.write('hamLa: %s\n' % mean_hamLa)
            file.write('accSb: %s\n' % mean_accSb)
            file.write('hamLb: %s\n' % mean_hamLb)
        
        fold += 1
        sess.close()
        
    ######################################################################################
    # Write final results to file
    ######################################################################################
    with open("Transformer_results.txt", "a") as file:
        file.write("Supervised, total\n")
        file.write('Test loss over folds: %s\n' % (fold_meanLoss/folds))
        file.write('meanF1a over folds: %s\n' % (fold_meanF1a/folds))
        file.write('accSa: %s\n' % (fold_mean_accSa/folds))
        file.write('hamLa: %s\n' % (fold_mean_hamLa/folds))
        file.write('meanF1b over folds: %s\n' % (fold_meanF1b/folds))
        file.write('accSb: %s\n' % (fold_mean_accSb/folds))
        file.write('hamLb: %s\n' % (fold_mean_hamLb/folds))
    
    print("\nFinal results supervised:")
    print('Test loss over folds: %s' % (fold_meanLoss/folds))
    print('meanF1a over folds: %s' % (fold_meanF1a/folds))
    print('accSa over folds: %s' % (fold_mean_accSa/folds))
    print('hamLa over folds: %s' % (fold_mean_hamLa/folds))
    print('meanF1b over folds: %s' % (fold_meanF1b/folds))
    print('accSb over folds: %s' % (fold_mean_accSb/folds))
    print('hamLb over folds: %s\n' % (fold_mean_hamLb/folds))
        
    del prediction_model, sc, x_anomalous, y_anomalous
    gc.collect()
    
    return pred
    
class Forecasting_Attention:
    def __init__(self, embedded_size, learning_rate, input_size, output_size,
                 num_blocks, num_heads, future_day=1, min_freq = 50, dropout_rate = 0.8):
        self.X = tf.placeholder(tf.float32, (None, input_size, output_size))
        self.Y = tf.placeholder(tf.float32, (None, future_day, output_size))
        
        encoder_embedded = tf.layers.dense(self.X, embedded_size)
        encoder_embedded = tf.nn.dropout(encoder_embedded, keep_prob = dropout_rate)
        x_mean = tf.reduce_mean(self.X, axis = 2)
        en_masks = tf.sign(x_mean)
        encoder_embedded += sinusoidal_position_encoding(self.X, en_masks, embedded_size)
        
        for i in range(num_blocks):
            with tf.variable_scope('encoder_self_attn_%d'%i,reuse=tf.AUTO_REUSE):
                encoder_embedded = multihead_attn(queries = encoder_embedded,
                                                       keys = encoder_embedded,
                                                       q_masks = en_masks,
                                                       k_masks = en_masks,
                                                       future_binding = False,
                                                       num_units = embedded_size,
                                                       num_heads = num_heads)
            
            if i == num_blocks-1: self.attention_outputs = tf.identity(encoder_embedded)
                
            with tf.variable_scope('encoder_feedforward_%d'%i,reuse=tf.AUTO_REUSE):
                encoder_embedded = pointwise_feedforward(encoder_embedded,
                                                         embedded_size,
                                                         activation = tf.nn.relu)
                
            if i == num_blocks-1: self.feedforward_outputs = tf.identity(encoder_embedded)
                        
        encoder_embedded = tf.transpose(tf.layers.dense(encoder_embedded, output_size), perm=[0,2,1])
        self.logits = tf.transpose(tf.layers.dense(encoder_embedded, future_day), perm=[0,2,1])
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.Y - self.logits)))
        self.indiv_max_rse = tf.sqrt(tf.reduce_max(tf.square(self.Y - self.logits), axis=0))
        self.optimizer = RAdamOptimizer(learning_rate=learning_rate, 
                                        total_steps=10000, 
                                        warmup_proportion=0.1, 
                                        min_lr=1e-7).minimize(self.rmse)
        
class Prediction_Attention:
    def __init__(self, input_size, output_size, embedded_size, learning_rate, num_blocks, num_heads,\
                 batch_size, future_day=1, loss_weight=1, min_freq = 50, dropout_rate = 0.8):
        self.EncoderOutput = tf.placeholder(tf.float32, (batch_size, input_size, embedded_size))
        self.Xtrue = tf.placeholder(tf.float32, (batch_size, future_day, output_size))
        self.Xanom = tf.placeholder(tf.float32, (batch_size, future_day, output_size))
        self.Yanom = tf.placeholder(tf.float32, (batch_size, future_day, output_size))
    
        #Prepare encoder output
        encoder_embedded = tf.layers.dense(self.EncoderOutput, embedded_size)
        encoder_embedded = tf.nn.dropout(encoder_embedded, keep_prob = dropout_rate)
        
        # Prepare difference between forecast and actual measurement
        Xdiff = tf.math.sqrt(tf.math.square(tf.math.subtract(self.Xtrue, self.Xanom)))        
        anom_embedded = tf.layers.dense(Xdiff, embedded_size)
        anom_embedded = tf.nn.dropout(anom_embedded, keep_prob = dropout_rate)
        
        # Concatenate
        anom_encoder_embedded = tf.concat([encoder_embedded, anom_embedded], 1)
        x_anom_mean = tf.reduce_mean(anom_encoder_embedded, axis = 2)
        en_anom_masks = tf.sign(x_anom_mean)
        anom_encoder_embedded += sinusoidal_position_encoding(anom_encoder_embedded, en_anom_masks, embedded_size)
        
        self.attention_outputs = [tf.Variable(np.empty((batch_size, input_size+future_day, embedded_size), dtype=np.float32))] * num_blocks
        self.feedforward_outputs = [tf.Variable(np.empty((batch_size, input_size+future_day, embedded_size), dtype=np.float32))] * num_blocks
        for i in range(num_blocks):
            with tf.variable_scope('encoder_self_attn_%d'%i,reuse=tf.AUTO_REUSE):
                anom_encoder_embedded = multihead_attn(queries = anom_encoder_embedded,
                                                       keys = anom_encoder_embedded,
                                                       q_masks = en_anom_masks,
                                                       k_masks = en_anom_masks,
                                                       future_binding = False,
                                                       num_units = embedded_size,
                                                       num_heads = num_heads)

            self.attention_outputs[i] = self.attention_outputs[i].assign(tf.identity(anom_encoder_embedded))
                
            with tf.variable_scope('encoder_feedforward_%d'%i,reuse=tf.AUTO_REUSE):
                anom_encoder_embedded = pointwise_feedforward(anom_encoder_embedded,
                                                              embedded_size,
                                                              activation = tf.nn.relu)

            self.feedforward_outputs[i] = self.feedforward_outputs[i].assign(tf.identity(anom_encoder_embedded))
                
        anom_encoder_embedded = tf.transpose(tf.layers.dense(anom_encoder_embedded, output_size), perm=[0,2,1])
        self.logits = tf.transpose(tf.layers.dense(anom_encoder_embedded, future_day), perm=[0,2,1])
        self.loss = tf.nn.weighted_cross_entropy_with_logits(labels=self.Yanom, logits=self.logits, pos_weight=loss_weight)
        self.optimizer = RAdamOptimizer(learning_rate=learning_rate, 
                                        total_steps=10000, 
                                        warmup_proportion=0.1, 
                                        min_lr=1e-7).minimize(self.loss)
