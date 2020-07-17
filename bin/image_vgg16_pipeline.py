#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# 

"""
Created on Sun Apr  2 10:33:22 2017, Feb/2020

@author: Firoj Alam, Ferda Ofli
"""

# Load VGGNet
import os
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.applications.vgg16 import preprocess_input
import pandas as pd
from keras.optimizers import SGD
import numpy as np
import warnings
import datetime
import optparse
import os, errno
import performance as performance
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn import preprocessing
import keras.callbacks as callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, CSVLogger, TensorBoard
from time import time
from keras.optimizers import SGD, Adam

class ImgInstance(object):
    def __init__(self,id=1,imgpath="",label=""):
        self.id = id
        self.imgpath = imgpath        
        self.label = label    
        
def vgg_model(num_class):
    vgg16 = VGG16(weights='imagenet')
    fc2 = vgg16.get_layer('fc2').output
    prediction = Dense(output_dim=num_class, activation='softmax', name='predictions')(fc2)
    model = Model(input=vgg16.input, output=prediction)
    print(model.summary())
    # Freeze All Layers Except Bottleneck Layers for Fine-Tuning
#    for layer in model.layers:
#        if layer.name in ['fc1', 'last_layer_output', 'logit']:
#            continue
#        layer.trainable = False
    # Compile with SGD Optimizer and a Small Learning Rate
    adam = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
    # sgd = SGD(lr=1e-4, momentum=0.9)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Set Up Data Generators
# Note the use of the preprocessing_function argument in keras.preprocessing.image.ImageDataGenerator

def preprocess_input_vgg(x):
    """Wrapper around keras.applications.vgg16.preprocess_input()
    to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.
    
    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)
    
    Note we cannot pass keras.applications.vgg16.preprocess_input()
    directly to to keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument because the former expects a
    4D tensor whereas the latter expects a 3D tensor. Hence the
    existence of this wrapper.
    
    Returns a numpy 3darray (the preprocessed image).
    
    """
    
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X, data_format=None, mode='torch')
    return X[0]
    
    
def check_dir(directory):    
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise    
            
    
def generate_data_file(data_file,delim="\t"):
    
    data_list=[]
    label_list=[]
    id_list=[]
    with open(data_file, 'rU') as f:
        next(f)    
        for line in f:
            line = line.strip()   
            if (line==""):
                continue                            		
            row=line.split(delim)
            tweet_id=row[0]            
            img_path=row[4]
            label=row[7]
            data_list.append(img_path)
            label_list.append(label)
            id_list.append(tweet_id)
    image_len = len(data_list)
    all_images = np.empty([image_len, 224, 224, 3])
    all_labels = []
    for i in range(image_len):
           img = image.load_img(data_list[i], target_size=(224, 224))
           img = image.img_to_array(img)
           img = np.expand_dims(img, axis=0)
           img = preprocess_input(img)        
           lab = label_list[i]        
           all_images[i, :, :, :] = img
           all_labels.append (lab)
    num_class=len(set(all_labels))   
    print("num classes: "+str(num_class))    
    le = preprocessing.LabelEncoder()
    y=le.fit_transform(all_labels) 
    y=np.asarray(y)
    all_labels = to_categorical(y, num_classes=num_class)
    print(all_labels.shape)    
    return all_images,all_labels,id_list,le,num_class;

def dir_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

"""
It assumes the inputs are text files, train, development and test. 
"""
if __name__ == "__main__":
    warnings.filterwarnings("ignore")    
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="train_data")
    parser.add_option('-v', action="store", dest="val_data")
    parser.add_option('-t', action="store", dest="test_data")
    parser.add_option('-m', action="store", dest="model_file")
    parser.add_option('-o', action="store", dest="outputfile")
    parser.add_option("--patience", action="store", dest="patience", default=10, type="int")
    parser.add_option("--patience-lr", action="store", dest="patience_lr", default=5, type="int")
    parser.add_option("-c", "--checkpoint_log", action="store", dest="checkpoint_log", default="./checkpoint_log/",
                      type="string")


    options, args = parser.parse_args()
    a = datetime.datetime.now().replace(microsecond=0)    
    batch_size = 16
    nb_epoch=100

    train_data,train_label,train_id,train_le,num_class=generate_data_file(options.train_data)
    nb_train_samples=len(train_data)
    val_data,val_label,val_id,_,_=generate_data_file(options.val_data)
    nb_validation_samples=len(val_data)
    test_data,test_label,test_id,_,_=generate_data_file(options.test_data)
    nb_test_samples=len(test_data)

    best_model_path=options.model_file  
    out_file=options.outputfile
    outFile=open(out_file,"w")

    
    ######## Data input ########        
    train_data=preprocess_input_vgg(train_data)
    val_data=preprocess_input_vgg(val_data)

    ######## Load vgg pre-trained model ########        
    model=vgg_model(num_class);
    
    ######## Do Fine-Tuning ########        
    patience_early_stop = options.patience
    patience_learning_rate = options.patience
    log_path = options.checkpoint_log
    dir_exist(log_path)
    log_dir = os.path.abspath(os.path.dirname(log_path))
    dir_exist(log_dir)
    base_name = os.path.basename(options.train_data)
    base_name = os.path.splitext(base_name)[0]
    log_file = log_dir + "/" + base_name + "_log.txt"


    callback = callbacks.EarlyStopping(monitor='val_acc', patience=patience_early_stop, verbose=1, mode='max')
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=patience_learning_rate, verbose=1,
                                                factor=0.1, min_lr=0.0001,mode='max')

    tensorboard = TensorBoard(log_dir=log_dir + "/{}".format(time()), write_graph=True, write_images=False,
                              batch_size=batch_size, write_grads=True, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None)

    csv_logger = CSVLogger(log_file, append=False, separator='\t')
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_acc', verbose=1,
                                 save_best_only=True, save_weights_only=True, mode='max')
    callbacks_list = [callback, learning_rate_reduction, csv_logger, checkpoint]


    model.fit(train_data, train_label, batch_size=batch_size, epochs=nb_epoch,verbose=1, validation_data=(val_data, val_label),callbacks=callbacks_list)    
                         
    ######## Save the model ########    
    print ("saving model...")
    #model.save(best_model_path)
    
   ##### Evaluation

    val_prob=model.predict([val_data], batch_size=batch_size, verbose=1)
    AUC, accu, P, R, F1, report = performance.performance_measure_cnn(val_label, val_prob, train_le)

    result = str("{0:.4f}".format(accu)) + "\t" + str("{0:.4f}".format(P)) + "\t" + str(
        "{0:.4f}".format(R)) + "\t" + str("{0:.4f}".format(F1))+ "\t" + str("{0:.4f}".format(AUC)) + "\n"
    print(result)
    print (report)
    outFile.write(options.val_data + "\n")
    outFile.write(result)
    outFile.write(report)
    val_data=[]
    test_data = preprocess_input_vgg(test_data)
    test_prob=model.predict([test_data], batch_size=batch_size, verbose=1)
    AUC, accu, P, R, F1, report = performance.performance_measure_cnn(test_label, test_prob, train_le)

    result = str("{0:.4f}".format(accu)) + "\t" + str("{0:.4f}".format(P)) + "\t" + str(
        "{0:.4f}".format(R)) + "\t" + str("{0:.4f}".format(F1))+ "\t" + str("{0:.4f}".format(AUC)) + "\n"

    print(result)
    print (report)
    outFile.write(options.test_data+"\n")
    outFile.write(result)
    outFile.write(report)

        
    b = datetime.datetime.now().replace(microsecond=0)
    print "time taken:"
    print(b-a)    
