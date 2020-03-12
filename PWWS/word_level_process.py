# -*- coding: utf-8 -*-
import spacy
import os
import re
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
from .read_files import read_imdb_files, read_yahoo_files, read_agnews_files
from .config import config

nlp = spacy.load('en_core_web_sm')

global imdb_tokenizer,yahoo_tokenizer,agnews_tokenizer
imdb_tokenizer=yahoo_tokenizer=agnews_tokenizer=None


def get_tokenizer(dataset):

    texts = None
    if dataset == 'imdb':
        global imdb_tokenizer
        if imdb_tokenizer is not None:
            return imdb_tokenizer
        texts, _ = read_imdb_files('train')
        imdb_tokenizer = Tokenizer(num_words=config.num_words[dataset])
        imdb_tokenizer.fit_on_texts(texts)
        return imdb_tokenizer
    elif dataset == 'yahoo':
        global yahoo_tokenizer
        if yahoo_tokenizer is not None:
            return yahoo_tokenizer
        texts, _, _ = read_yahoo_files()
        yahoo_tokenizer = Tokenizer(num_words=config.num_words[dataset])
        yahoo_tokenizer.fit_on_texts(texts)
        return yahoo_tokenizer
    elif dataset == 'agnews':
        global agnews_tokenizer
        if agnews_tokenizer is not None:
            return agnews_tokenizer
        texts, _, _ = read_agnews_files('train')
        agnews_tokenizer = Tokenizer(num_words=config.num_words[dataset])
        agnews_tokenizer.fit_on_texts(texts)
        return agnews_tokenizer



def word_process(train_texts, train_labels, test_texts, test_labels, dataset):
    maxlen = config.word_max_len[dataset]
    tokenizer = get_tokenizer(dataset)

    x_train_seq = tokenizer.texts_to_sequences(train_texts)
    x_test_seq = tokenizer.texts_to_sequences(test_texts)
    x_train = sequence.pad_sequences(x_train_seq, maxlen=maxlen, padding='post', truncating='post')
    x_test = sequence.pad_sequences(x_test_seq, maxlen=maxlen, padding='post', truncating='post')
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    return x_train, y_train, x_test, y_test


def text_to_vector(text, tokenizer, dataset):
    maxlen = config.word_max_len[dataset]
    vector = tokenizer.texts_to_sequences([text])
    vector = sequence.pad_sequences(vector, maxlen=maxlen, padding='post', truncating='post')
    return vector


def text_to_vector_for_all(text_list, tokenizer, dataset):
    maxlen = config.word_max_len[dataset]
    vector = tokenizer.texts_to_sequences(text_list)
    vector = sequence.pad_sequences(vector, maxlen=maxlen, padding='post', truncating='post')
    return vector

