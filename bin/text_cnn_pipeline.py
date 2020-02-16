#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Sun Apr  2 10:33:22 2017

@author: Firoj Alam
"""
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import numpy as np
import sys
from cnn import data_process as data_process
from cnn import cnn_filter as cnn_filter
from gensim.models import KeyedVectors
from keras.layers import Dense, Input, Dropout, Activation, Flatten
from keras.models import Sequential,Model
import keras.callbacks as callbacks
from keras.callbacks import ModelCheckpoint
import cnn.performance as performance
import os, errno
from keras.models import model_from_yaml
import pickle
import zipfile
from keras.models import load_model
import warnings
import datetime
import optparse



seed = 1337
np.random.seed(seed)

def save_model(model, model_file_name, tokenizer,label_encoder):

    model_file=model_file_name+".hdf5"
    tokenizer_file=model_file_name+".tokenizer"
    label_encoder_file=model_file_name+".label_encoder"

    configfile=model_file_name+".config"
    configFile=open(configfile,"w")
    configFile.write("word_vec_model_file="+model_file+"\n")
    configFile.write("tokenizer_file="+tokenizer_file+"\n")
    configFile.write("label_encoder_file="+label_encoder_file+"\n")
    configFile.close()
    
    files=[]
    files.append(configfile)
    
    # serialize weights to HDF5
    model.save(model_file)
    files.append(model_file)
    
    # saving tokenizer
    with open(tokenizer_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    files.append(tokenizer_file)
    
    # saving label_encoder
    with open(label_encoder_file, 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    files.append(label_encoder_file)
    
    
if __name__ == '__main__':    
    warnings.filterwarnings("ignore")    
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="train_data")
    parser.add_option('-v', action="store", dest="val_data")
    parser.add_option('-t', action="store", dest="test_data")
    parser.add_option('-m', action="store", dest="word_vec_model_file")
    parser.add_option('-o', action="store", dest="outputfile")    
    options, args = parser.parse_args()
    a = datetime.datetime.now().replace(microsecond=0)
    
    trainFile=options.train_data
    devFile=options.val_data
    tstFile=options.test_data
    resultsFile=options.outputfile
    best_model_path=options.model_file    
    outFile=open(resultsFile,"w")
    
    MAX_SEQUENCE_LENGTH = 30
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 300
    batch_size=20    
    nb_epoch=200

    ######## Word-Embedding ########                        
    #modelFile="/Users/firojalam/QCRI/w2v/GoogleNews-vectors-negative300.txt"    
    #modelFile="/export/home/fialam/crisis_semi_supervised/crisis-tweets/model/crisis_word_vector.txt"
    modelFile="/export/home/fialam/w2v_models/crisis_word_vector.txt"
    #modelFile = "/data/w2v_models/crisis_word_vector.txt"
    emb_model = KeyedVectors.load_word2vec_format(modelFile, binary=False)
    #emb_model = ""
    ######## Data input ########                    
    delim="\t"
    train_x,train_y,train_le,train_labels,word_index,tokenizer=data_process.getTrDataMM(trainFile,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH,delim) 
    dev_x,dev_y,dev_le,dev_labels,_=data_process.getDevDataMM(devFile,tokenizer,MAX_SEQUENCE_LENGTH,delim)                
    test_x,test_y,test_le,test_labels,_=data_process.getDevDataMM(tstFile,tokenizer,MAX_SEQUENCE_LENGTH,delim)
    print("Train: "+str(len(train_x)))
    
    y_true=np.argmax(train_y, axis = 1)
    y_true=train_le.inverse_transform(y_true)
    nb_classes=len(set(y_true.tolist()))
    print ("Number of classes: "+str(nb_classes))

    ######## Text text_network ########                
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    cnn = cnn_filter.text_cnn(emb_model,word_index,MAX_NB_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,inputs)            
    callback = callbacks.EarlyStopping(monitor='val_acc',patience=30,verbose=0, mode='max')            
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [callback,checkpoint]        
    R,C=train_x.shape
    network=Dense(800)(cnn)
    network=Activation('relu')(network)
    network=Dense(500)(network)
    network=Activation('relu')(network)
    #text_network=Dropout(0.02)(text_network)
    out=Dense(nb_classes, activation='softmax')(network)
    model=Model(inputs=inputs,outputs=out)
    model.compile(loss='categorical_crossentropy',optimizer='Adadelta',metrics=['accuracy'])
    model.fit([train_x], train_y, batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=([dev_x], dev_y),callbacks=callbacks_list)

    ######## Save the model ########
    save_model(model,best_model_path,tokenizer,train_le)
    model = load_model(best_model_path)

    ######## Evaluation ########    
    dev_prob=model.predict([dev_x], batch_size=batch_size, verbose=1)
    test_prob=model.predict([test_x], batch_size=batch_size, verbose=1)    

    ##### Dev    
    accu,P,R,F1,wAUC,AUC,report=performance.performance_measure_tf(dev_y,dev_prob,dev_le,dev_labels,devFile)
    accu=accu*100
    wauc=wAUC*100
    auc=AUC*100
    precision=P*100
    recall=R*100
    f1_score=F1*100
    result=str("{0:.2f}".format(auc))+"\t"+str("{0:.2f}".format(accu))+"\t"+str("{0:.2f}".format(precision))+"\t"+str("{0:.2f}".format(recall))+"\t"+str("{0:.2f}".format(f1_score))+"\n"
    print(result)
    print (report)
    outFile.write(tstFile+"\n")
    outFile.write(result)
    outFile.write(report)
    
    ###### Test
    accu,P,R,F1,wAUC,AUC,report=performance.performance_measure_tf(test_y,test_prob,test_le,test_labels,tstFile)
    accu=accu*100
    wauc=wAUC*100
    auc=AUC*100
    precision=P*100
    recall=R*100
    f1_score=F1*100
    result=str("{0:.2f}".format(auc))+"\t"+str("{0:.2f}".format(accu))+"\t"+str("{0:.2f}".format(precision))+"\t"+str("{0:.2f}".format(recall))+"\t"+str("{0:.2f}".format(f1_score))+"\n"    
    print(result)
    print (report)
    outFile.write(devFile+"\n")
    outFile.write(result)
    outFile.write(report)
    