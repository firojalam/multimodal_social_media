# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 20:42:24 2017; Feb/2020

@author: Firoj Alam
"""

import numpy as np

np.random.seed(1337)  # for reproducibility

import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
from sklearn import preprocessing
import random
import aidrtokenize as aidrtokenize

random.seed(1337)


def file_exist(file_name):
    if os.path.exists(file_name):
        return True
    else:
        return False


def read_stop_words(file_name):
    if (not file_exist(file_name)):
        print("Please check the file for stop words, it is not in provided location " + file_name)
        sys.exit(0)
    stop_words = []
    with open(file_name, 'rU') as f:
        for line in f:
            line = line.strip()
            if (line == ""):
                continue
            stop_words.append(line)
    return stop_words;


stop_words_file = "bin/etc/stop_words_english.txt"
stop_words = read_stop_words(stop_words_file)


def read_train_data(dataFile, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, delim):
    """
    Prepare the data
    """
    data = []
    labels = []
    with open(dataFile, 'rb') as f:
        next(f)
        for line in f:
            line = line.decode(encoding='utf-8', errors='strict')
            line = line.strip()
            if (line == ""):
                continue
            row = line.split(delim)
            txt = row[3].strip().lower()
            txt = aidrtokenize.tokenize(txt)
            label = row[6]
            if (len(txt) < 1):
                print (txt)
                continue
            data.append(txt)
            labels.append(label)

    data_shuf = []
    lab_shuf = []
    index_shuf = list(range(len(data)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        data_shuf.append(data[i])
        lab_shuf.append(labels[i])

    le = preprocessing.LabelEncoder()
    yL = le.fit_transform(lab_shuf)
    labels = list(le.classes_)

    label = yL.tolist()
    yC = len(set(label))
    yR = len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y = np.array(y, dtype=np.int32)

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token="OOV_TOK")
    tokenizer.fit_on_texts(data_shuf)
    sequences = tokenizer.texts_to_sequences(data_shuf)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    print('Shape of data tensor:', data.shape)
    return data, y, le, labels, word_index, tokenizer


def read_dev_data(dataFile, tokenizer, MAX_SEQUENCE_LENGTH, delim, train_le):
    """
    Prepare the data
    """
    id_list=[]
    data = []
    labels = []
    with open(dataFile, 'rb') as f:
        next(f)
        for line in f:
            line = line.decode(encoding='utf-8', errors='strict')
            line = line.strip()
            if (line == ""):
                continue
            row = line.split(delim)
            t_id= row[2].strip().lower()
            txt = row[3].strip().lower()
            txt = aidrtokenize.tokenize(txt)
            # txt = remove_stop_words(txt, stop_words)
            label = row[6]
            if (len(txt) < 1):
                print (txt)
                continue
            # if(isinstance(txt, str)):
            data.append(txt)
            labels.append(label)
            id_list.append(t_id)

    print(len(data))
    data_shuf = []
    lab_shuf = []
    index_shuf = list(range(len(data)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        data_shuf.append(data[i])
        lab_shuf.append(labels[i])

    le = train_le  # preprocessing.LabelEncoder()
    yL = le.transform(labels)
    labels = list(le.classes_)

    label = yL.tolist()
    yC = len(set(label))
    yR = len(label)
    y = np.zeros((yR, yC))
    y[np.arange(yR), yL] = 1
    y = np.array(y, dtype=np.int32)

    sequences = tokenizer.texts_to_sequences(data)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data.shape)
    return data, y, le, labels, word_index,id_list


def load_embedding(fileName):
    print('Indexing word vectors.')
    embeddings_index = {}
    f = open(fileName)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index;


def prepare_embedding(word_index, model, MAX_NB_WORDS, EMBEDDING_DIM):
    # prepare embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM), dtype=np.float32)
    print(len(embedding_matrix))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        try:
            if(word in model):
                embedding_vector = model[word][0:EMBEDDING_DIM]  # embeddings_index.get(word)
                embedding_matrix[i] = np.asarray(embedding_vector, dtype=np.float32)
            else:
                rng = np.random.RandomState()
                embedding_vector = rng.randn(EMBEDDING_DIM) #np.random.random(num_features)
                #embedding_vector = np.zeros(EMBEDDING_DIM)  # np.random.random(num_features)
                embedding_matrix[i] = np.asarray(embedding_vector,dtype=np.float32)
        except KeyError:
            try:
                print(word.encode('utf-8') +" not found... assigning random")
                rng = np.random.RandomState()
                embedding_vector = rng.randn(EMBEDDING_DIM) #np.random.random(num_features)
                #embedding_vector = np.zeros(EMBEDDING_DIM)  # np.random.random(num_features)
                embedding_matrix[i] = np.asarray(embedding_vector,dtype=np.float32)
            except KeyError:
                continue
    return embedding_matrix;


def str_to_indexes(s):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    input_size = 1014
    length = input_size
    alphabet_size = len(alphabet)
    char_dict = {}  # Maps each character to an integer
    # self.no_of_classes = num_of_classes
    for idx, char in enumerate(alphabet):
        char_dict[char] = idx + 1
    length = input_size

    """
    Convert a string to character indexes based on character dictionary.

    Args:
        s (str): String to be converted to indexes

    Returns:
        str2idx (np.ndarray): Indexes of characters in s

    """
    s = s.lower()
    max_length = min(len(s), length)
    str2idx = np.zeros(length, dtype='int64')
    for i in range(1, max_length + 1):
        c = s[-i]
        if c in char_dict:
            str2idx[i - 1] = char_dict[c]
    return str2idx
