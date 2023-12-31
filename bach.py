#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:41:42 2023

@author: sergio
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import os

def acc(pred,targ):
    a = 100 - np.linalg.norm(np.abs(pred - targ)) / np.linalg.norm(targ) * 100
    return a

def ThePhantomOfTheOpera(LL,nIT,trainPath,testPath):
    """
    LL -------- length of the note sequence input
    N --------- number of examples in the training set
    nIT ------- number of iterations for backprop
    trainPath - path to the training dataset
    testPath -- path to the testing dataset
    """
    
    L = LL
    DOR = 0.3 #drop out rate
    
    #TRAIN
    #build X, Y from training set
    csvfiles = [f for f in os.listdir(trainPath) if f.endswith(".csv")]
    
    dataFrames = []

    for csvfile in csvfiles:
        filePath = os.path.join(trainPath, csvfile)
        dataFrame = pd.read_csv(filePath)
        dataFrames.append(dataFrame)

    minRows = min(df.shape[0] for df in dataFrames)
    if minRows < L+1:
        L = minRows-1
        print('')
        str = f'note sequences not long enought, new length set to {L}'
        print(str)
    
    #chop chop chop
    choppedFrames = [df.iloc[:L+1] for df in dataFrames]
    
    xFrame = [df.iloc[:L] for df in choppedFrames]
    yFrame = [df.iloc[L:] for df in choppedFrames]
    
    X = np.array(xFrame)
    Y = np.array(yFrame)  
    
    N, L, nNotes = np.shape(X)
    
    phantom = tf.keras.Sequential([        
        tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(L, nNotes)),        
        tf.keras.layers.Dropout(DOR), 
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=False)),        
        tf.keras.layers.Dropout(DOR),                          
        tf.keras.layers.Dense(25, activation='relu'),                                    
        tf.keras.layers.Dense(nNotes, activation='linear')
    ])

    phantom.compile(
        loss='mean_squared_error',                       
        optimizer='RMSprop',                             
        metrics=['mean_squared_error']                   
    )
    YY = Y.reshape(N,nNotes)
    
    history = phantom.fit(x=X, y=YY, epochs=nIT, batch_size=len(Y))
    H = phantom.predict(X)
    accTrain = acc(H,YY)

    #TEST
    #build X, Y from testing set
    csvfiles = [f for f in os.listdir(testPath) if f.endswith(".csv")]
    
    dataFrames = []

    for csvfile in csvfiles:
        filePath = os.path.join(testPath, csvfile)
        dataFrame = pd.read_csv(filePath)
        dataFrames.append(dataFrame)

    minRows = min(df.shape[0] for df in dataFrames)
    if minRows < L+1:
        L = minRows-1
        print('')
        str = f'note sequences not long enough, new length set to {L}'
        print(str)
    
    #chop chop chop
    choppedFrames = [df.iloc[:L+1] for df in dataFrames]
    
    xFrame = [df.iloc[:L] for df in choppedFrames]
    yFrame = [df.iloc[L:] for df in choppedFrames]
    
    X = np.array(xFrame)
    Y = np.array(yFrame)    
    
    N, L, nNotes = np.shape(X)
    YY = Y.reshape(N,nNotes)
    
    loss, error = phantom.evaluate(X, YY)
    H = phantom.predict(X)
    accTest = acc(H,YY)
    
    return loss, error, accTrain, accTest
