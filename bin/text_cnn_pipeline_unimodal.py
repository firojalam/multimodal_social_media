#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Sun Apr  2 10:33:22 2017; Feb/2020

@author: Firoj Alam
"""
import os


import numpy as np
import sys
import data_process_new as data_process
import cnn_filter as cnn_filter
from gensim.models import KeyedVectors
from keras.layers import Dense, Input, Dropout, Activation, Flatten, BatchNormalization
from keras.models import Sequential, Model
import keras.callbacks as callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, CSVLogger, TensorBoard
import keras
import performance as performance
import os, errno
import warnings
import datetime
import optparse
import pickle
from time import time
from datetime import datetime

seed = 1337
np.random.seed(seed)

def save_model(model, model_dir, model_file_name, tokenizer, label_encoder):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    base_name = os.path.basename(model_file_name)
    base_name = os.path.splitext(base_name)[0]
    timestr = datetime.now().strftime("%d-%m-%Y_%I-%M-%S")

    model_file = model_dir + "/" + base_name +"_"+timestr+ ".hdf5"
    tokenizer_file = model_dir + "/" + base_name +"_"+timestr+ ".tokenizer"
    label_encoder_file = model_dir + "/" + base_name +"_"+timestr+ ".label_encoder"

    configfile = model_dir + "/" + base_name + ".config"
    configFile = open(configfile, "w")
    configFile.write("model_file=" + model_file + "\n")
    configFile.write("tokenizer_file=" + tokenizer_file + "\n")
    configFile.write("label_encoder_file=" + label_encoder_file + "\n")
    configFile.close()

    files = []
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


def file_exist(w2v_checkpoint):
    if os.path.exists(w2v_checkpoint):
        return True
    else:
        return False


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    #a = datetime.now().replace(microsecond=0)
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="train_data")
    parser.add_option('-v', action="store", dest="val_data")
    parser.add_option('-t', action="store", dest="test_data")
    parser.add_option('-m', action="store", dest="model_file")
    parser.add_option('-l', action="store", dest="label_file")
    parser.add_option('-o', action="store", dest="output_file")
    parser.add_option("--w2v_checkpoint", action="store", dest="w2v_checkpoint", default=None, type="string")
    parser.add_option("--log_dir", action="store", dest="log_dir", default="./log", type="string")
    parser.add_option("--log_file", action="store", dest="log_file", default="./checkpoint_log/log.txt", type="string")
    parser.add_option("--checkpoint_log", action="store", dest="checkpoint_log", default="./checkpoint_log",
                      type="string")
    parser.add_option("--vocab_size", action="store", dest="vocab_size", default=20000, type="int")
    parser.add_option("--embedding_dim", action="store", dest="embedding_dim", default=300, type="int")
    parser.add_option("--batch_size", action="store", dest="batch_size", default=128, type="int")
    parser.add_option("--nb_epoch", action="store", dest="nb_epoch", default=1000, type="int")
    parser.add_option("--max_seq_length", action="store", dest="max_seq_length", default=25, type="int")
    parser.add_option("--patience", action="store", dest="patience", default=100, type="int")
    parser.add_option("--patience-lr", action="store", dest="patience_lr", default=10, type="int")

    options, args = parser.parse_args()
    a = datetime.now().replace(microsecond=0)

    train_file = options.train_data
    dev_file = options.val_data
    test_file = options.test_data
    best_model_path = options.model_file
    out_label_file_name = options.label_file
    results_file = options.output_file
    out_file = open(results_file, "w")

    log_path = options.checkpoint_log
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(options.checkpoint_log):
        os.makedirs(options.checkpoint_log)

    log_file = options.log_file
    log_path = options.checkpoint_log
    log_dir = os.path.abspath(os.path.dirname(log_file))
    base_name = os.path.basename(log_file)
    timestr = datetime.now().strftime("%d-%m-%Y_%I-%M-%S")
    log_file = log_dir + "/" + base_name +"_"+timestr+ "_text_log.txt"

    MAX_SEQUENCE_LENGTH = options.max_seq_length


    ######## Data input ########                    
    delim = "\t"
    train_x, train_y, train_le, train_labels, word_index, tokenizer = data_process.read_train_data(train_file,
                                                                                                   options.vocab_size,
                                                                                                   MAX_SEQUENCE_LENGTH,
                                                                                                   delim)
    dev_x, dev_y, dev_le, dev_labels, _,id_list = data_process.read_dev_data(dev_file, tokenizer, MAX_SEQUENCE_LENGTH, delim,train_le)
    test_x, test_y, test_le, test_labels, _,id_list = data_process.read_dev_data(test_file, tokenizer, MAX_SEQUENCE_LENGTH,delim,train_le)
    print("Train: " + str(len(train_x)))

    y_true = np.argmax(train_y, axis=1)
    y_true = train_le.inverse_transform(y_true)
    nb_classes = len(set(y_true.tolist()))
    print ("Number of classes: " + str(nb_classes))

    ######## Word-Embedding ########
    if (options.w2v_checkpoint and file_exist(options.w2v_checkpoint)):
        options.emb_matrix = pickle.load(open(options.w2v_checkpoint, "rb"))
    else:
        model_file = "/export/home/fialam/w2v_models/crisis_word_vector.txt"
        emb_model = KeyedVectors.load_word2vec_format(model_file, binary=False)
        embedding_matrix = data_process.prepare_embedding(word_index, emb_model, options.vocab_size,
                                                          options.embedding_dim)
        print("Embedding size: " + str(embedding_matrix.shape))
        options.emb_matrix = embedding_matrix
        options.vocab_size, options.embedding_dim = embedding_matrix.shape
        pickle.dump(options.emb_matrix, open(options.w2v_checkpoint, "wb"))

    ######## Text network ########                
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    # cnn = cnn_filter.text_cnn(options.emb_matrix, word_index, options.vocab_size, options.embedding_dim,
    #                           MAX_SEQUENCE_LENGTH, inputs)
    cnn = cnn_filter.kimCNN(options.emb_matrix, word_index, options.vocab_size, options.embedding_dim,
                              MAX_SEQUENCE_LENGTH, inputs)



    callback = callbacks.EarlyStopping(monitor='val_acc', patience=options.patience, verbose=0, mode='max')
    tensorboard = TensorBoard(log_dir=options.checkpoint_log + "/{}".format(time()), histogram_freq=0, write_graph=True,
                              write_images=True, embeddings_freq=0, embeddings_layer_names="Embedding layer",
                              embeddings_metadata=None)
    csv_logger = CSVLogger(log_file, append=False, separator='\t')

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=options.patience_lr, verbose=1, factor=0.01,
                                                min_lr=0.00001)
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')

    R, C = train_x.shape
    network = Activation('relu')(cnn)
    network = Dropout(0.02)(network)
    network = Dense(100)(network)
    network = Activation('relu')(network)
    network = Dense(50)(network)
    network = Activation('relu')(network)

    out = Dense(nb_classes, activation='softmax',name='lrec-softmax')(network)
    model = Model(inputs=inputs, outputs=out)
    lr = 0.00001
    print("lr= "+str(lr)+", beta_1=0.9, beta_2=0.999, amsgrad=False")
    adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])



    callbacks_list = [callback, learning_rate_reduction, tensorboard, csv_logger, checkpoint]

    print(model.summary())
    history = model.fit([train_x], train_y, batch_size=options.batch_size,  epochs=options.nb_epoch, verbose=1,validation_data=([dev_x], dev_y), callbacks=callbacks_list)

    ######## Save the model ########
    # save_model(model, best_model_path, tokenizer, train_le)
    # model = load_model(best_model_path+".hdf5")
    ######## Save the model ########
    print ("saving model...")
    model.load_weights(best_model_path)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print ("Best saved model loaded...")

    dir_name = os.path.dirname(best_model_path)
    base_name = os.path.basename(train_file)
    base_name = os.path.splitext(base_name)[0]
    model_dir = dir_name + "/" + base_name + "_text"
    save_model(model, model_dir, best_model_path, tokenizer, train_le)

    ######## Evaluation ########    
    dev_prob = model.predict([dev_x], batch_size=options.batch_size, verbose=1)
    test_prob = model.predict([test_x], batch_size=options.batch_size, verbose=1)

    ##### Dev
    dir_name = os.path.dirname(out_label_file_name)
    base_name = os.path.basename(out_label_file_name)
    base_name = os.path.splitext(base_name)[0]
    dev_out_label_file_name = dir_name + "/" + base_name + "_dev_labels.txt"

    AUC, accu, P, R, F1, report = performance.performance_measure_cnn(dev_y, dev_prob, train_le)
    result = str("{0:.4f}".format(accu)) + "\t" + str("{0:.4f}".format(P)) + "\t" + str(
        "{0:.4f}".format(R)) + "\t" + str("{0:.4f}".format(F1))+ "\t" + str("{0:.4f}".format(AUC)) + "\n"
    print(result)
    print (report)
    out_file.write(dev_file + "\n")
    out_file.write(result)
    out_file.write(report)

    ###### Test
    test_out_label_file_name = dir_name + "/" + base_name + "_test_labels.txt"
    AUC, accu, P, R, F1, report = performance.performance_measure_cnn(test_y, test_prob, train_le)
    result = str("{0:.4f}".format(accu)) + "\t" + str("{0:.4f}".format(P)) + "\t" + str(
        "{0:.2f}".format(R)) + "\t" + str("{0:.4f}".format(F1)) + "\t" + str("{0:.4f}".format(AUC))+ "\n"
    print("results-cnn:\t"+base_name+"\t"+result)
    print (report)
    out_file.write( test_file+ "\n")
    out_file.write(result)
    out_file.write(report)

    conf_mat_str = performance.format_conf_mat(test_y, test_prob, train_le)
    out_file.write(conf_mat_str+"\n")
    out_file.close()
    b = datetime.now().replace(microsecond=0)
    print ("time taken:")
    print(b - a)