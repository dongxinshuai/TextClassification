# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import argparse
import os
import numpy as np
from .read_files import split_imdb_files, split_yahoo_files, split_agnews_files
from .word_level_process import word_process, get_tokenizer
from .char_level_process import char_process
from .adversarial_tools import ForwardGradWrapper, ForwardGradWrapper_pytorch, adversarial_paraphrase
import time
from .unbuffered import Unbuffered

sys.stdout = Unbuffered(sys.stdout)

def write_origin_input_texts(origin_input_texts_path, test_texts, test_samples_cap=None):
    if test_samples_cap is None:
        test_samples_cap = len(test_texts)
    with open(origin_input_texts_path, 'a') as f:
        for i in range(test_samples_cap):
            f.write(test_texts[i] + '\n')


def fool_text_classifier_pytorch(device,model, dataset='imdb'):
    clean_samples_cap = 20
    print('clean_samples_cap:', clean_samples_cap)

    # get tokenizer
    tokenizer = get_tokenizer(dataset)

    # Read data set
    x_test = y_test = None
    test_texts = None
    if dataset == 'imdb':
        train_texts, train_labels, test_texts, test_labels = split_imdb_files()
        x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

    elif dataset == 'agnews':
        train_texts, train_labels, test_texts, test_labels = split_agnews_files()
        x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

    elif dataset == 'yahoo':
        train_texts, train_labels, test_texts, test_labels = split_yahoo_files()
        x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)

    grad_guide = ForwardGradWrapper_pytorch(model, device)
    classes_prediction = grad_guide.predict_classes(x_test[: clean_samples_cap])

    print('Crafting adversarial examples...')
    successful_perturbations = 0
    failed_perturbations = 0
    sub_rate_list = []
    NE_rate_list = []

    start_cpu = time.clock()
    fa_path = r'./fool_result/{}'.format(dataset)
    if not os.path.exists(fa_path):
        os.makedirs(fa_path)
    adv_text_path = r'./fool_result/{}/adv_{}.txt'.format(dataset, str(clean_samples_cap))
    change_tuple_path = r'./fool_result/{}/change_tuple_{}.txt'.format(dataset, str(clean_samples_cap))
    #file_1 = open(adv_text_path, "a")
    #file_2 = open(change_tuple_path, "a")
    for index, text in enumerate(test_texts[: clean_samples_cap]):
        sub_rate = 0
        NE_rate = 0
        print('_____{}______.'.format(index))
        if np.argmax(y_test[index]) == classes_prediction[index]:
            print('do')
            # If the ground_true label is the same as the predicted label
            adv_doc, adv_y, sub_rate, NE_rate, change_tuple_list = adversarial_paraphrase(input_text=text,
                                                                                          true_y=np.argmax(y_test[index]),
                                                                                          grad_guide=grad_guide,
                                                                                          tokenizer=tokenizer,
                                                                                          dataset=dataset,
                                                                                          level='word')
            if adv_y != np.argmax(y_test[index]):
                successful_perturbations += 1
                print('{}. Successful example crafted.'.format(index))
            else:
                failed_perturbations += 1
                print('{}. Failure.'.format(index))

            text = adv_doc
            sub_rate_list.append(sub_rate)
            NE_rate_list.append(NE_rate)
            #file_2.write(str(index) + str(change_tuple_list) + '\n')
        #file_1.write(text + " sub_rate: " + str(sub_rate) + "; NE_rate: " + str(NE_rate) + "\n")
    end_cpu = time.clock()
    print('CPU second:', end_cpu - start_cpu)

    #mean_sub_rate = sum(sub_rate_list) / len(sub_rate_list)
    #mean_NE_rate = sum(NE_rate_list) / len(NE_rate_list)
    print('substitution:', sum(sub_rate_list))
    print('sum substitution:', len(sub_rate_list))
    print('NE rate:', sum(NE_rate_list))
    print('sum NE:', len(NE_rate_list))
    print("succ attack %d"%(successful_perturbations))
    print("fail attack %d"%(failed_perturbations))
    #file_1.close()
    #file_2.close()


