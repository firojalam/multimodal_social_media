# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 20:42:24 2017; Feb/2020

@author: firojalam
"""


import numpy as np
# for reproducibility
seed = 1337
np.random.seed(seed)

import numpy as np
np.random.seed(1337)
from keras.layers import Conv1D, MaxPooling1D, Embedding
import shlex
from subprocess import Popen, PIPE  
from collections import Counter
import random
from keras.layers import concatenate
from keras.constraints import max_norm
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, MaxPooling2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate

def get_exitcode_stdout_stderr(cmd):
    """
    Execute the external command and get its exitcode, stdout and stderr.
    """
    args = shlex.split(cmd)

    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    #
    return exitcode, out, err
    
def label_one_hot(yL):
    label=yL.tolist()
    yC=len(set(label))
    yR=len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y=np.array(y,dtype=np.int32)
    return y  
    
def upsampling(train_x,train_y):
    ########## Upsampling    
    y_true=np.argmax(train_y, axis = 1)
    smote = ""#SMOTE(ratio=0.5, kind='borderline1',n_jobs=5)
    X_resampled, y_resampled = smote.fit_sample(train_x,y_true)
  
    ########## Shuffling  
    combined = list(zip(X_resampled, y_resampled))
    random.shuffle(combined)
    X_resampled[:], y_resampled[:] = zip(*combined)
    y_resampled_true=label_one_hot(y_resampled)
    dimension = X_resampled.shape[1]
    y_resampled_true=label_one_hot(y_resampled)
    print(len(X_resampled))
    X_resampled=np.array(X_resampled)
    print(X_resampled.shape)    
    counts = Counter(y_resampled)
    print(counts)   
    return X_resampled, y_resampled_true, dimension
  
def text_cnn(embedding_matrix,word_index,MAX_NB_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,inputs):
    nb_words = min(MAX_NB_WORDS, len(word_index)+1)
    embedding_layer=Embedding(output_dim=EMBEDDING_DIM, input_dim=nb_words, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,trainable=True)(inputs)
    # embedding_layer=Embedding(output_dim=EMBEDDING_DIM, input_dim=nb_words, input_length=MAX_SEQUENCE_LENGTH,trainable=False)(inputs)

    ########## CNN: Filtering with Max pooling:
    #nb_filter = 250
    #filter_length = 3
    branches = [] # models to be merged
    filter_window_sizes=[2,3,4,5]
    pool_size=2
    num_filters=[100,150,200,300]
    for filter_len,nb_filter in zip(filter_window_sizes,num_filters):
        branch = embedding_layer
        branch=Conv1D(filters=nb_filter,
                                 kernel_size=int(filter_len),
                                 padding='valid',
                                 activation='relu',
                                 strides=1,
                                 kernel_initializer='glorot_uniform',
                                 kernel_constraint=max_norm(3), bias_constraint=max_norm(3))(branch)
        branch=MaxPooling1D(pool_size=pool_size)(branch)
        branch=Flatten()(branch)
        branches.append(branch)
    merged_model=concatenate(branches)

    return merged_model


def text_cnn_2d(embedding_matrix,word_index,MAX_NB_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,inputs):
    # this returns a tensor
    print("Creating Model...")
    nb_words = min(MAX_NB_WORDS, len(word_index)+1)
    embedding=Embedding(output_dim=EMBEDDING_DIM, input_dim=nb_words, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,trainable=True)(inputs)
    # embedding = Embedding(input_dim=nb_words, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(inputs)

    reshape = Reshape((MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,1))(embedding)
    filter_window_sizes = [2, 3, 4]
    # filter_sizes = [3, 4, 5]
    num_filters = 512
    conv_0 = Conv2D(num_filters, kernel_size=(filter_window_sizes[0], EMBEDDING_DIM), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_window_sizes[1], EMBEDDING_DIM), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_window_sizes[2], EMBEDDING_DIM), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_window_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_window_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_window_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    merged_model = Flatten()(concatenated_tensor)
    return merged_model



## sentence CNN by Y.Kim
def kimCNN(embedding_matrix,word_index,MAX_NB_WORDS,EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,sequence_input):
    """
    Convolution neural network model for sentence classification.
    Parameters
    ----------
    EMBEDDING_DIM: Dimension of the embedding space.
    MAX_SEQUENCE_LENGTH: Maximum length of the sentence.
    MAX_NB_WORDS: Maximum number of words in the vocabulary.
    embeddings_index: A dict containing words and their embeddings.
    word_index: A dict containing words and their indices.
    labels_index: A dict containing the labels and their indices.
    Returns
    -------
    compiled keras model
    """
    print('Preparing embedding matrix.')
    # num_words = min(MAX_NB_WORDS, len(word_index))
    # nb_words = min(MAX_NB_WORDS, len(word_index) + 1)
    # embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    # for word, i in word_index.items():
    #     if i >= MAX_NB_WORDS:
    #         continue
    #     embedding_vector = embeddings_index.get(word)
    #     if embedding_vector is not None:
    #         # words not found in embedding index will be all-zeros.
    #         embedding_matrix[i] = embedding_vector

    # embedding_layer = Embedding(nb_words,
    #                             EMBEDDING_DIM,
    #                             weights=[embedding_matrix],
    #                             input_length=MAX_SEQUENCE_LENGTH,
    #                             trainable=True)
    nb_words = min(MAX_NB_WORDS, len(word_index)+1)
    embedding_layer = Embedding(output_dim=EMBEDDING_DIM, input_dim=nb_words, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,trainable=True)



    print('Training model.')

    # sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    print(embedded_sequences.shape)


    # add first conv filter
    embedded_sequences = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(embedded_sequences)
    x = Conv2D(300, (5, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    x = MaxPool2D((MAX_SEQUENCE_LENGTH - 5 + 1, 1))(x)


    # add second conv filter.
    y = Conv2D(300, (4, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    y = MaxPool2D((MAX_SEQUENCE_LENGTH - 4 + 1, 1))(y)


    # add third conv filter.
    z = Conv2D(300, (3, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    z = MaxPool2D((MAX_SEQUENCE_LENGTH - 3 + 1, 1))(z)

    # add third conv filter.
    z1 = Conv2D(300, (2, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    z1 = MaxPool2D((MAX_SEQUENCE_LENGTH - 2 + 1, 1))(z1)

    # add third conv filter.
    w1 = Conv2D(300, (1, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    w1 = MaxPool2D((MAX_SEQUENCE_LENGTH - 1 + 1, 1))(w1)
    # concate the conv layers
    # alpha = concatenate([x,y,z,z1])
    alpha = concatenate([w1,z1,z,y])

    # flatted the pooled features.
    merged_model = Flatten()(alpha)

    return merged_model