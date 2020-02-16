#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# 

"""
Created on Sun Apr  2 10:33:22 2017; Feb/2020

@author: Firoj Alam, Ferda Ofli
Adopted from:
# https://nbviewer.jupyter.org/gist/embanner/6149bba89c174af3bfd69537b72bca74
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    
"""

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
import warnings
import datetime
import optparse
import os, errno
import performance as performance
import keras.callbacks as callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, CSVLogger, TensorBoard
from gensim.models import KeyedVectors
from keras.layers import Input, Activation, Add, Concatenate, Dropout
import data_process_multimodal_pair as data_process
import cnn_filter as cnn_filter
from keras.models import load_model
from keras.layers import concatenate
from time import time
import pickle
from keras.layers.normalization import BatchNormalization
from crisis_data_generator_image_optimized import DataGenerator
import keras
from keras.applications.resnet50 import ResNet50


class ImgInstance(object):
    def __init__(self, id=1, imgpath="", label=""):
        self.id = id
        self.imgpath = imgpath
        self.label = label

def resnet_model():
    IMG_HEIGHT=224
    IMG_WIDTH=224
    restnet = ResNet50(include_top=False, weights='imagenet', input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))
    # for layer in restnet.layers: # in case if we want to freeze the
    #     layer.trainable = False
    last_layer_output = restnet.layers[-1].output
    last_layer_output = keras.layers.Flatten()(last_layer_output)
    return last_layer_output,restnet;

def vgg_model():
    vgg16 = VGG16(weights='imagenet')
    # Freeze All Layers Except Bottleneck Layers for Fine-Tuning
    # for layer in vgg16.layers:
    #     if layer.name in ['fc1', 'fc2', 'logit']:
    #         continue
    #     layer.trainable = False
    last_layer_output = vgg16.get_layer('fc2').output
    # vgg16.summary()
    return last_layer_output, vgg16


def check_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def file_exist(w2v_checkpoint):
    if os.path.exists(w2v_checkpoint):
        return True
    else:
        return False


def save_model(model, model_dir, model_file_name, tokenizer, label_encoder):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    base_name = os.path.basename(model_file_name)
    base_name = os.path.splitext(base_name)[0]
    model_file = model_dir + "/" + base_name + ".hdf5"
    tokenizer_file = model_dir + "/" + base_name + ".tokenizer"
    label_encoder_file = model_dir + "/" + base_name + ".label_encoder"

    configfile = model_dir + "/" + base_name + "_v2.config"
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


def write_results(out_file, file_name, accu, P, R, F1, wAUC, AUC, report, conf_mat):
    accu = accu * 100
    wauc = wAUC * 100
    auc = AUC * 100
    precision = P * 100
    recall = R * 100
    f1_score = F1 * 100
    result = str("{0:.2f}".format(auc)) + "\t" + str("{0:.2f}".format(accu)) + "\t" + str(
        "{0:.2f}".format(precision)) + "\t" + str("{0:.2f}".format(recall)) + "\t" + str(
        "{0:.2f}".format(f1_score)) + "\n"
    print(result)
    print (report)
    out_file.write(file_name + "\n")
    out_file.write(result)
    out_file.write(report)
    out_file.write(conf_mat)

def dir_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

"""
It assumes the inputs are text files, train, development and test. 
"""
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="train_data", default=None, type="string")
    parser.add_option('-v', action="store", dest="val_data", default=None, type="string")
    parser.add_option('-t', action="store", dest="test_data", default=None, type="string")
    parser.add_option('-m', action="store", dest="model_file", default="best_model.hdf5", type="string")
    parser.add_option('-o', action="store", dest="outputfile", default="results.tsv", type="string")
    parser.add_option("-w", "--w2v_checkpoint", action="store", dest="w2v_checkpoint",
                      default="w2v_models/data_w2v_info.model", type="string")
    parser.add_option("-d", "--log_dir", action="store", dest="log_dir", default="model_log/", type="string")
    # parser.add_option("-l","--log_file", action="store", dest="log_file", default="./log", type="string")
    parser.add_option("-c", "--checkpoint_log", action="store", dest="checkpoint_log", default="./checkpoint_log/",
                      type="string")
    parser.add_option("-x", "--vocab_size", action="store", dest="vocab_size", default=20000, type="int")
    parser.add_option("--embedding_dim", action="store", dest="embedding_dim", default=300, type="int")
    parser.add_option("--batch_size", action="store", dest="batch_size", default=32, type="int")
    parser.add_option("--nb_epoch", action="store", dest="nb_epoch", default=1000, type="int")
    parser.add_option("--max_seq_length", action="store", dest="max_seq_length", default=25, type="int")
    parser.add_option("--patience", action="store", dest="patience", default=100, type="int")
    parser.add_option("--patience-lr", action="store", dest="patience_lr", default=10, type="int")
    #parser.add_option("-n", "--num_of_inst", action="store", dest="num_of_inst", default=10, type="int")
    parser.add_option("--text_sim_score", action="store", dest="text_sim_score", default=0.6, type="float")
    parser.add_option("--image_sim_score", action="store", dest="image_sim_score", default=0.6, type="float")
    parser.add_option("--total_sim_score", action="store", dest="total_sim_score", default=0.6, type="float")
    parser.add_option("--label_index", action="store", dest="label_index", default=6, type="int")
    parser.add_option("--image_dump", action="store", dest="image_dump", default="data/task_data/all_images_data_dump.npy", type="string")

    options, args = parser.parse_args()
    a = datetime.datetime.now().replace(microsecond=0)

    train_file = options.train_data
    val_file = options.val_data
    test_file = options.test_data
    out_file = options.outputfile
    best_model_path = options.model_file
    log_path = options.checkpoint_log
    log_dir = os.path.abspath(os.path.dirname(log_path))
    dir_exist(log_dir)

    base_name = os.path.basename(train_file)
    base_name = os.path.splitext(base_name)[0]
    log_file = log_dir + "/" + base_name + "_log_v2.txt"

    with open(options.image_dump, 'rb') as handle:
        images_npy_data = pickle.load(handle)

    #print(images_npy_data.keys())
    ######## Parameters ########
    MAX_SEQUENCE_LENGTH = options.max_seq_length
    MAX_NB_WORDS = options.vocab_size
    EMBEDDING_DIM = options.embedding_dim
    batch_size = options.batch_size
    nb_epoch = options.nb_epoch
    patience_early_stop = options.patience
    patience_learning_rate = options.patience
    dir_exist(options.checkpoint_log)
    delim = "\t"

    #### training dataset
    dir_name = os.path.dirname(train_file)
    base_name = os.path.basename(train_file)
    base_name = os.path.splitext(base_name)[0]

    train_x, train_image_list, train_y, train_le, train_labels, word_index, tokenizer = data_process.read_train_data_multimodal(
        train_file,
        MAX_NB_WORDS,
        MAX_SEQUENCE_LENGTH,int(options.label_index),
        delim)


    #### development dataset
    base_name = os.path.basename(val_file)
    base_name = os.path.splitext(base_name)[0]


    dev_x, dev_image_list, dev_y, dev_le, dev_labels, _ = data_process.read_dev_data_multimodal(val_file,
                                                                                                tokenizer,
                                                                                                MAX_SEQUENCE_LENGTH,int(options.label_index),
                                                                                                delim)

    nb_classes = len(set(train_labels))
    print ("Number of classes: " + str(nb_classes))
    params = {"max_seq_length": MAX_SEQUENCE_LENGTH, "batch_size": batch_size,
              "n_classes": nb_classes, "shuffle": True}
    train_data_generator = DataGenerator(train_image_list, train_x, images_npy_data, train_y, **params)

    params = {"max_seq_length": MAX_SEQUENCE_LENGTH, "batch_size": batch_size,
              "n_classes": nb_classes, "shuffle": False}
    val_data_generator = DataGenerator(dev_image_list, dev_x, images_npy_data, dev_y, **params)

    MAX_SEQUENCE_LENGTH = options.max_seq_length
    MAX_NB_WORDS = options.vocab_size


    ######## Word-Embedding ########
    if (options.w2v_checkpoint and file_exist(options.w2v_checkpoint)):
        options.emb_matrix = pickle.load(open(options.w2v_checkpoint, "rb"))
    else:
        word_vec_model_file = "/home/local/QCRI/fialam/w2v_models/crisis_word_vector.txt"
        emb_model = KeyedVectors.load_word2vec_format(word_vec_model_file, binary=False)
        embedding_matrix = data_process.prepare_embedding(word_index, emb_model, options.vocab_size,
                                                          options.embedding_dim)
        options.emb_matrix = embedding_matrix
        options.vocab_size, options.embedding_dim = embedding_matrix.shape
        pickle.dump(options.emb_matrix, open(options.w2v_checkpoint, "wb"))

    ######## Text text_network ########
    print("Embedding size: " + str(options.emb_matrix.shape))
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))

    cnn = cnn_filter.kimCNN(options.emb_matrix, word_index, options.vocab_size, options.embedding_dim,
                            MAX_SEQUENCE_LENGTH, inputs)
    # R, C = train_x.shape
    text_network = Dense(1000, activation='relu')(cnn)
    text_network = BatchNormalization()(text_network)

    ######## Image text_network ########
    last_layer_output, vgg16 = vgg_model()


    last_layer_output = Dense(1000, activation='relu')(last_layer_output)
    last_layer_output = BatchNormalization()(last_layer_output)


    ######## Merge image and text networks ########
    merged_network = concatenate([last_layer_output, text_network], axis=-1)
    merged_network = BatchNormalization()(merged_network)
    merged_network = Dropout(0.4)(merged_network)
    merged_network = Dense(500, activation='relu')(merged_network)
    merged_network = Dropout(0.2)(merged_network)
    merged_network = Dense(100, activation='relu')(merged_network)
    merged_network = Dropout(0.02)(merged_network)
    out = Dense(nb_classes, activation='softmax')(merged_network)
    model = Model(inputs=[vgg16.input, inputs], outputs=out)


    lr = 0.00001
    print("lr= "+str(lr)+", beta_1=0.9, beta_2=0.999, amsgrad=False")
    adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    print(model.summary())

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
    history = model.fit_generator(generator=train_data_generator, epochs=nb_epoch, validation_data=val_data_generator,
                                  use_multiprocessing=True,
                                  workers=4, verbose=1, callbacks=callbacks_list)

    ######## Save the model ########
    print ("saving model...")
    model.load_weights(best_model_path)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print ("Best saved model loaded...")

    dir_name = os.path.dirname(best_model_path)
    base_name = os.path.basename(train_file)
    base_name = os.path.splitext(base_name)[0]
    model_dir = dir_name + "/" + base_name
    save_model(model, model_dir, best_model_path, tokenizer, train_le)

    ############ Test data  ########
    dir_name = os.path.dirname(out_file)
    base_name = os.path.basename(out_file)
    base_name = os.path.splitext(base_name)[0]
    out_file_name = dir_name + "/" + base_name + ".txt"
    out_file = open(out_file_name, "w")

    dev_prob = model.predict_generator(val_data_generator, verbose=1)
    print("dev true len: "+str(len(dev_y)))
    print("dev pred len: " + str(len(dev_prob)))
    AUC, accu, P, R, F1, report = performance.performance_measure_cnn(dev_y, dev_prob, train_le)

    result = str("{0:.4f}".format(accu)) + "\t" + str("{0:.4f}".format(P)) + "\t" + str(
        "{0:.4f}".format(R)) + "\t" + str("{0:.4f}".format(F1))+ "\t" + str("{0:.4f}".format(AUC)) + "\n"
    print(result)
    print (report)
    out_file.write(val_file + "\n")
    out_file.write(result)
    out_file.write(report)

    test_x, test_image_list, test_y, test_le, test_labels, ids = data_process.read_dev_data_multimodal(test_file,
                                                                                                       tokenizer,
                                                                                                       MAX_SEQUENCE_LENGTH,int(options.label_index),
                                                                                                       delim)

    print ("Number of classes: " + str(nb_classes))
    params = {"max_seq_length": MAX_SEQUENCE_LENGTH, "batch_size": batch_size,
              "n_classes": nb_classes, "shuffle": False}
    print("image size: "+str(len(test_image_list)))
    print("test x: "+str(len(test_x)))
    print("test y: "+str(len(test_y)))
    test_data_generator = DataGenerator(test_image_list, test_x, images_npy_data, test_y, **params)

    ######## Evaluation ########
    test_prob = model.predict_generator(test_data_generator, verbose=1)
    print("test true len: "+str(len(test_y)))
    print("test pred len: " + str(len(test_prob)))

    AUC, accu, P, R, F1, report = performance.performance_measure_cnn(dev_y, dev_prob, train_le)
    result = str("{0:.4f}".format(accu)) + "\t" + str("{0:.4f}".format(P)) + "\t" + str(
        "{0:.4f}".format(R)) + "\t" + str("{0:.4f}".format(F1))+ "\t" + str("{0:.4f}".format(AUC)) + "\n"
    print(result)
    print (report)
    out_file.write(val_file + "\n")
    out_file.write(result)
    out_file.write(report)


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


    #write_results(out_file, test_file, accu, P, R, F1, wAUC, AUC, report, conf_mat)
    out_file.close()

    b = datetime.datetime.now().replace(microsecond=0)
    print ("time taken:")
    print(b - a)
