#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm
import sklearn
from sklearn.metrics import f1_score

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

from tensorflow.keras.utils import HDF5Matrix
from keras_radam.training import RAdamOptimizer

def mtsTransformer(modelID, fn, L1, L2, sign=21, noLayers=1, noUnits=512, noHeads=16, batch_size=16, epochs=300):
    
    print('Loading Data')
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
    # Parameters
    ######################################################################################
    num_blocks = noLayers
    num_heads = noHeads
    size_layer = noUnits
    size_embedding = noUnits
    future_day = 30
    learning_rate = 1e-4
    tf.reset_default_graph()

    modelnn = Attention(size_layer, size_embedding, learning_rate, x_train.shape[1],\
                        y_train.shape[2], num_blocks, num_heads, future_day)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    ######################################################################################
    # Training loop
    ######################################################################################
    pbar = tqdm(range(epochs), desc = 'train loop')
    prev_rmse, counter = 1e3, 0
    for i in pbar:
        total_rmse = []
        max_rse = np.zeros((future_day,sign))
        for k in range(batch_size, x_train.shape[0], batch_size):
            batch_x = x_train[k-batch_size:k, :, :]
            batch_y = y_train[k-batch_size:k]
            logits, _, rmse, indiv_max_rse = sess.run(
                [modelnn.logits, modelnn.optimizer, modelnn.rmse, modelnn.indiv_max_rse],
                feed_dict = {
                    modelnn.X: batch_x,
                    modelnn.Y: batch_y
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
            batch_x = x_val[l-batch_size:l, :, :]
            out_logits = sess.run(
                modelnn.logits,
                feed_dict = {
                    modelnn.X: batch_x,
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
            batch_x = x_test[k-batch_size:k, :, :]
            out_logits = sess.run(
                modelnn.logits,
                feed_dict = {
                    modelnn.X: batch_x,
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
    pbar = tqdm(range(batch_size, x_anomalous.shape[0], batch_size), desc = "anom pred")
    anom_rmse = 0
    anom_indiv_max_rse = np.zeros(x_anomalous.shape)
    for k in pbar:
        batch_pred = np.zeros((batch_size, x_anomalous.shape[1], x_anomalous.shape[2]))
        batch_pred[:,:L1,:] = x_anomalous[k-batch_size:k,:L1,:]
        out_logits = sess.run(
            modelnn.logits,
            feed_dict = {
                modelnn.X:batch_pred[:,:L1,:]
            },
        )
        anom_rmse += np.sqrt(np.mean(np.square(x_anomalous[k-batch_size:k,L1:] - out_logits)))
        anom_indiv_max_rse[k-batch_size:k,L1:] = np.sqrt(np.max(np.square(x_anomalous[k-batch_size:k,L1:] - out_logits),axis=0))
        batch_pred[:,L1:,:] = out_logits
        x_anompred[k-batch_size:k,:,:] = batch_pred
    
    print("Mean RMSE anomaly forecasting:", anom_rmse/(x_anomalous.shape[0]/batch_size))
    print("Individual Max RSE anomaly forecasting:", np.max(np.max(anom_indiv_max_rse, axis=0), axis=0))
    sess.close()
    
    print("Signal anomaly prediction:")
    labels = np.amax(y_anomalous[:,L1:,:sign-1], axis=1)
    max_rse_prediction = np.sqrt(np.max(np.square(x_anomalous[:,L1:,:sign-1] - x_anompred[:,L1:,:sign-1]),axis=1))
    predictions = np.zeros(max_rse_prediction.shape)
    for idx, sample in enumerate(max_rse_prediction):
        predictions[idx] = (sample >= [x for x in np.max(thresholds, axis=0)[:sign-1]])
    
    meanF1 = 0
    for i in range(sign-1):
        meanF1 += sklearn.metrics.f1_score(labels[:,i], predictions[:,i], average='weighted')
        print("F1 signal",i,sklearn.metrics.f1_score(labels[:,i], predictions[:,i], average='weighted'))
    meanF1a = meanF1/(sign-1)
    print("Mean F1 signal",meanF1a)
    
    print("Signal x Ts anomaly prediction:")
    labels = y_anomalous[:,L1:,:sign-1]
    max_rse_prediction = np.sqrt(np.square(x_anomalous[:,L1:,:sign-1] - x_anompred[:,L1:,:sign-1]))
    predictions = np.zeros(max_rse_prediction.shape)
    for idx, sample in enumerate(max_rse_prediction):
        for jdx, step in enumerate(sample):
            predictions[idx,jdx,:] = (step >= [x for x in thresholds[jdx,:sign-1]])
    
    meanF1 = 0
    for i in range(sign-1):
        meanF1 += sklearn.metrics.f1_score(labels[:,:,i], predictions[:,:,i], average='weighted')
        print("F1 signal",i,sklearn.metrics.f1_score(labels[:,:,i], predictions[:,:,i], average='weighted'))
    meanF1b = meanF1/(sign-1)
    print("Mean F1 signal",meanF1b)

    # Write results to file
    with open("Transformer_results.txt", "a") as file:
        file.write('test metrics model %s:\n' % modelID)
        file.write("%s\n" % (test_rmse/(x_test.shape[0]/batch_size)))
        for item in np.max(np.max(test_indiv_max_rse, axis=0), axis=0):
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
    
class Attention:
    def __init__(self, size_layer, embedded_size, learning_rate, input_size, output_size,
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
                                             num_units = size_layer,
                                             num_heads = num_heads)

            with tf.variable_scope('encoder_feedforward_%d'%i,reuse=tf.AUTO_REUSE):
                encoder_embedded = pointwise_feedforward(encoder_embedded,
                                                    embedded_size,
                                                    activation = tf.nn.relu)
                
        encoder_embedded = tf.transpose(tf.layers.dense(encoder_embedded, output_size), perm=[0,2,1])
        self.logits = tf.transpose(tf.layers.dense(encoder_embedded, future_day), perm=[0,2,1])
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.Y - self.logits)))
        self.indiv_max_rse = tf.sqrt(tf.reduce_max(tf.square(self.Y - self.logits), axis=0))
        self.optimizer = RAdamOptimizer(learning_rate=learning_rate, 
                               total_steps=10000, 
                               warmup_proportion=0.1, 
                               min_lr=1e-7).minimize(self.rmse)
        
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

if __name__ == '__main__':
    ### Seed 1
    mtsTransformer('m1', 'synthetic_L1_300_L2_30_SIG_20_SEED_77', 300, 30, 21, 1, 512, 16)
    mtsTransformer('m2', 'synthetic_L1_300_L2_30_SIG_20_SEED_77', 300, 30, 21, 1, 512, 32)
    mtsTransformer('m3', 'synthetic_L1_300_L2_30_SIG_20_SEED_77', 300, 30, 21, 1, 1024, 16)
    mtsTransformer('m4', 'synthetic_L1_300_L2_30_SIG_20_SEED_77', 300, 30, 21, 1, 1024, 32)
    mtsTransformer('m5', 'synthetic_L1_300_L2_30_SIG_20_SEED_77', 300, 30, 21, 2, 512, 16)
    mtsTransformer('m6', 'synthetic_L1_300_L2_30_SIG_20_SEED_77', 300, 30, 21, 2, 512, 32)
    mtsTransformer('m7', 'synthetic_L1_300_L2_30_SIG_20_SEED_77', 300, 30, 21, 2, 1024, 16)
    mtsTransformer('m8', 'synthetic_L1_300_L2_30_SIG_20_SEED_77', 300, 30, 21, 2, 1024, 32)
    
    mtsTransformer('m9', 'synthetic_L1_300_L2_30_SIG_40_SEED_77', 300, 30, 41, 1, 512, 16)
    mtsTransformer('m10', 'synthetic_L1_300_L2_30_SIG_40_SEED_77', 300, 30, 41, 1, 512, 32)
    mtsTransformer('m11', 'synthetic_L1_300_L2_30_SIG_40_SEED_77', 300, 30, 41, 1, 1024, 16)
    mtsTransformer('m12', 'synthetic_L1_300_L2_30_SIG_40_SEED_77', 300, 30, 41, 1, 1024, 32)
    mtsTransformer('m13', 'synthetic_L1_300_L2_30_SIG_40_SEED_77', 300, 30, 41, 2, 512, 16)
    mtsTransformer('m14', 'synthetic_L1_300_L2_30_SIG_40_SEED_77', 300, 30, 41, 2, 512, 32)
    mtsTransformer('m15', 'synthetic_L1_300_L2_30_SIG_40_SEED_77', 300, 30, 41, 2, 1024, 16)
    mtsTransformer('m16', 'synthetic_L1_300_L2_30_SIG_40_SEED_77', 300, 30, 41, 2, 1024, 32)
    
    mtsTransformer('m17', 'synthetic_L1_600_L2_60_SIG_20_SEED_77', 600, 60, 21, 1, 512, 16)
    mtsTransformer('m18', 'synthetic_L1_600_L2_60_SIG_20_SEED_77', 600, 60, 21, 1, 512, 32)
    mtsTransformer('m19', 'synthetic_L1_600_L2_60_SIG_20_SEED_77', 600, 60, 21, 1, 1024, 16)
    mtsTransformer('m20', 'synthetic_L1_600_L2_60_SIG_20_SEED_77', 600, 60, 21, 1, 1024, 32)
    mtsTransformer('m21', 'synthetic_L1_600_L2_60_SIG_20_SEED_77', 600, 60, 21, 2, 512, 16)
    mtsTransformer('m22', 'synthetic_L1_600_L2_60_SIG_20_SEED_77', 600, 60, 21, 2, 512, 32)
    mtsTransformer('m23', 'synthetic_L1_600_L2_60_SIG_20_SEED_77', 600, 60, 21, 2, 1024, 16)
    mtsTransformer('m24', 'synthetic_L1_600_L2_60_SIG_20_SEED_77', 600, 60, 21, 2, 1024, 32)
    
    mtsTransformer('m25', 'synthetic_L1_600_L2_60_SIG_40_SEED_77', 600, 60, 41, 1, 512, 16)
    mtsTransformer('m26', 'synthetic_L1_600_L2_60_SIG_40_SEED_77', 600, 60, 41, 1, 512, 32)
    mtsTransformer('m27', 'synthetic_L1_600_L2_60_SIG_40_SEED_77', 600, 60, 41, 1, 1024, 16)
    mtsTransformer('m28', 'synthetic_L1_600_L2_60_SIG_40_SEED_77', 600, 60, 41, 1, 1024, 32)
    mtsTransformer('m29', 'synthetic_L1_600_L2_60_SIG_40_SEED_77', 600, 60, 41, 2, 512, 16)
    mtsTransformer('m30', 'synthetic_L1_600_L2_60_SIG_40_SEED_77', 600, 60, 41, 2, 512, 32)
    mtsTransformer('m31', 'synthetic_L1_600_L2_60_SIG_40_SEED_77', 600, 60, 41, 2, 1024, 16)
    mtsTransformer('m32', 'synthetic_L1_600_L2_60_SIG_40_SEED_77', 600, 60, 41, 2, 1024, 32)
    ### Seed 2
    mtsTransformer('m33', 'synthetic_L1_300_L2_30_SIG_20_SEED_63', 300, 30, 21, 1, 512, 16)
    mtsTransformer('m34', 'synthetic_L1_300_L2_30_SIG_20_SEED_63', 300, 30, 21, 1, 512, 32)
    mtsTransformer('m35', 'synthetic_L1_300_L2_30_SIG_20_SEED_63', 300, 30, 21, 1, 1024, 16)
    mtsTransformer('m36', 'synthetic_L1_300_L2_30_SIG_20_SEED_63', 300, 30, 21, 1, 1024, 32)
    mtsTransformer('m37', 'synthetic_L1_300_L2_30_SIG_20_SEED_63', 300, 30, 21, 2, 512, 16)
    mtsTransformer('m38', 'synthetic_L1_300_L2_30_SIG_20_SEED_63', 300, 30, 21, 2, 512, 32)
    mtsTransformer('m39', 'synthetic_L1_300_L2_30_SIG_20_SEED_63', 300, 30, 21, 2, 1024, 16)
    mtsTransformer('m40', 'synthetic_L1_300_L2_30_SIG_20_SEED_63', 300, 30, 21, 2, 1024, 32)
    
    mtsTransformer('m41', 'synthetic_L1_300_L2_30_SIG_40_SEED_63', 300, 30, 41, 1, 512, 16)
    mtsTransformer('m42', 'synthetic_L1_300_L2_30_SIG_40_SEED_63', 300, 30, 41, 1, 512, 32)
    mtsTransformer('m43', 'synthetic_L1_300_L2_30_SIG_40_SEED_63', 300, 30, 41, 1, 1024, 16)
    mtsTransformer('m44', 'synthetic_L1_300_L2_30_SIG_40_SEED_63', 300, 30, 41, 1, 1024, 32)
    mtsTransformer('m45', 'synthetic_L1_300_L2_30_SIG_40_SEED_63', 300, 30, 41, 2, 512, 16)
    mtsTransformer('m46', 'synthetic_L1_300_L2_30_SIG_40_SEED_63', 300, 30, 41, 2, 512, 32)
    mtsTransformer('m47', 'synthetic_L1_300_L2_30_SIG_40_SEED_63', 300, 30, 41, 2, 1024, 16)
    mtsTransformer('m48', 'synthetic_L1_300_L2_30_SIG_40_SEED_63', 300, 30, 41, 2, 1024, 32)
    
    mtsTransformer('m49', 'synthetic_L1_600_L2_60_SIG_20_SEED_63', 600, 60, 21, 1, 512, 16)
    mtsTransformer('m50', 'synthetic_L1_600_L2_60_SIG_20_SEED_63', 600, 60, 21, 1, 512, 32)
    mtsTransformer('m51', 'synthetic_L1_600_L2_60_SIG_20_SEED_63', 600, 60, 21, 1, 1024, 16)
    mtsTransformer('m52', 'synthetic_L1_600_L2_60_SIG_20_SEED_63', 600, 60, 21, 1, 1024, 32)
    mtsTransformer('m53', 'synthetic_L1_600_L2_60_SIG_20_SEED_63', 600, 60, 21, 2, 512, 16)
    mtsTransformer('m54', 'synthetic_L1_600_L2_60_SIG_20_SEED_63', 600, 60, 21, 2, 512, 32)
    mtsTransformer('m55', 'synthetic_L1_600_L2_60_SIG_20_SEED_63', 600, 60, 21, 2, 1024, 16)
    mtsTransformer('m56', 'synthetic_L1_600_L2_60_SIG_20_SEED_63', 600, 60, 21, 2, 1024, 32)

    mtsTransformer('m57', 'synthetic_L1_600_L2_60_SIG_40_SEED_63', 600, 60, 41, 1, 512, 16)
    mtsTransformer('m58', 'synthetic_L1_600_L2_60_SIG_40_SEED_63', 600, 60, 41, 1, 512, 32)
    mtsTransformer('m59', 'synthetic_L1_600_L2_60_SIG_40_SEED_63', 600, 60, 41, 1, 1024, 16)
    mtsTransformer('m60', 'synthetic_L1_600_L2_60_SIG_40_SEED_63', 600, 60, 41, 1, 1024, 32)
    mtsTransformer('m61', 'synthetic_L1_600_L2_60_SIG_40_SEED_63', 600, 60, 41, 2, 512, 16)
    mtsTransformer('m62', 'synthetic_L1_600_L2_60_SIG_40_SEED_63', 600, 60, 41, 2, 512, 32)
    mtsTransformer('m63', 'synthetic_L1_600_L2_60_SIG_40_SEED_63', 600, 60, 41, 2, 1024, 16)
    mtsTransformer('m64', 'synthetic_L1_600_L2_60_SIG_40_SEED_63', 600, 60, 41, 2, 1024, 32)
    ### Seed 3
    mtsTransformer('m65', 'synthetic_L1_300_L2_30_SIG_20_SEED_12', 300, 30, 21, 1, 512, 16)
    mtsTransformer('m66', 'synthetic_L1_300_L2_30_SIG_20_SEED_12', 300, 30, 21, 1, 512, 32)
    mtsTransformer('m67', 'synthetic_L1_300_L2_30_SIG_20_SEED_12', 300, 30, 21, 1, 1024, 16)
    mtsTransformer('m68', 'synthetic_L1_300_L2_30_SIG_20_SEED_12', 300, 30, 21, 1, 1024, 32)
    mtsTransformer('m69', 'synthetic_L1_300_L2_30_SIG_20_SEED_12', 300, 30, 21, 2, 512, 16)
    mtsTransformer('m70', 'synthetic_L1_300_L2_30_SIG_20_SEED_12', 300, 30, 21, 2, 512, 32)
    mtsTransformer('m71', 'synthetic_L1_300_L2_30_SIG_20_SEED_12', 300, 30, 21, 2, 1024, 16)
    mtsTransformer('m72', 'synthetic_L1_300_L2_30_SIG_20_SEED_12', 300, 30, 21, 2, 1024, 32)
    
    mtsTransformer('m73', 'synthetic_L1_300_L2_30_SIG_40_SEED_12', 300, 30, 41, 1, 512, 16)
    mtsTransformer('m74', 'synthetic_L1_300_L2_30_SIG_40_SEED_12', 300, 30, 41, 1, 512, 32)
    mtsTransformer('m75', 'synthetic_L1_300_L2_30_SIG_40_SEED_12', 300, 30, 41, 1, 1024, 16)
    mtsTransformer('m76', 'synthetic_L1_300_L2_30_SIG_40_SEED_12', 300, 30, 41, 1, 1024, 32)
    mtsTransformer('m77', 'synthetic_L1_300_L2_30_SIG_40_SEED_12', 300, 30, 41, 2, 512, 16)
    mtsTransformer('m78', 'synthetic_L1_300_L2_30_SIG_40_SEED_12', 300, 30, 41, 2, 512, 32)
    mtsTransformer('m79', 'synthetic_L1_300_L2_30_SIG_40_SEED_12', 300, 30, 41, 2, 1024, 16)
    mtsTransformer('m80', 'synthetic_L1_300_L2_30_SIG_40_SEED_12', 300, 30, 41, 2, 1024, 32)
    
    mtsTransformer('m81', 'synthetic_L1_600_L2_60_SIG_20_SEED_12', 600, 60, 21, 1, 512, 16)
    mtsTransformer('m82', 'synthetic_L1_600_L2_60_SIG_20_SEED_12', 600, 60, 21, 1, 512, 32)
    mtsTransformer('m83', 'synthetic_L1_600_L2_60_SIG_20_SEED_12', 600, 60, 21, 1, 1024, 16)
    mtsTransformer('m84', 'synthetic_L1_600_L2_60_SIG_20_SEED_12', 600, 60, 21, 1, 1024, 32)
    mtsTransformer('m85', 'synthetic_L1_600_L2_60_SIG_20_SEED_12', 600, 60, 21, 2, 512, 16)
    mtsTransformer('m86', 'synthetic_L1_600_L2_60_SIG_20_SEED_12', 600, 60, 21, 2, 512, 32)
    mtsTransformer('m87', 'synthetic_L1_600_L2_60_SIG_20_SEED_12', 600, 60, 21, 2, 1024, 16)
    mtsTransformer('m88', 'synthetic_L1_600_L2_60_SIG_20_SEED_12', 600, 60, 21, 2, 1024, 32)

    mtsTransformer('m89', 'synthetic_L1_600_L2_60_SIG_40_SEED_12', 600, 60, 41, 1, 512, 16)
    mtsTransformer('m90', 'synthetic_L1_600_L2_60_SIG_40_SEED_12', 600, 60, 41, 1, 512, 32)
    mtsTransformer('m91', 'synthetic_L1_600_L2_60_SIG_40_SEED_12', 600, 60, 41, 1, 1024, 16)
    mtsTransformer('m92', 'synthetic_L1_600_L2_60_SIG_40_SEED_12', 600, 60, 41, 1, 1024, 32)
    mtsTransformer('m93', 'synthetic_L1_600_L2_60_SIG_40_SEED_12', 600, 60, 41, 2, 512, 16)
    mtsTransformer('m94', 'synthetic_L1_600_L2_60_SIG_40_SEED_12', 600, 60, 41, 2, 512, 32)
    mtsTransformer('m95', 'synthetic_L1_600_L2_60_SIG_40_SEED_12', 600, 60, 41, 2, 1024, 16)
    mtsTransformer('m96', 'synthetic_L1_600_L2_60_SIG_40_SEED_12', 600, 60, 41, 2, 1024, 32)