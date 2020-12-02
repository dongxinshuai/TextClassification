# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from six.moves import cPickle
import time,os,random
import itertools

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import NLLLoss,MultiLabelSoftMarginLoss,MultiLabelMarginLoss,BCELoss

from dataHelper_PWWS import load_synonyms_in_vocab, make_synthesized_iter_for_bert, snli_make_synthesized_iter
from PWWS.fool_pytorch import fool_text_classifier_pytorch, genetic_attack

import opts
import models
import utils

from solver.lr_scheduler import WarmupMultiStepLR

try:
    import cPickle as pickle
except ImportError:
    import pickle

#timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time()) ))
#performance_log_file =  os.path.join("log","result"+timeStamp+ ".csv") 
#if not os.path.exists(performance_log_file):
#    with open(performance_log_file,"w") as f:
#        f.write("argument\n")
#        f.close() 

def set_params(net, resume_model_path, data_parallel=False, bert=False):
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(resume_model_path), 'Error: ' + resume_model_path + 'checkpoint not found!'
    checkpoint = torch.load(resume_model_path)
    state_dict = checkpoint['net']
    from collections import OrderedDict
    sdict = OrderedDict()
    for key in state_dict.keys():
        new_key = key.split('module.')[-1]
        if data_parallel:
            new_key = 'module.' + new_key
        if bert:
            new_key = 'bert_model.' + new_key
        if 'hidden2label.weight' in new_key:
            new_key = 'hidden2label.weight'
        if 'hidden2label.bias' in new_key:
            new_key = 'hidden2label.bias'

        sdict[new_key]=state_dict[key]
    net.load_state_dict(sdict)
    return net


def test(opt, test_iter, syn_data, verbose=True):
    global_start= time.time()
    logger = utils.getLogger()
    model=models.setup(opt)

    from from_certified.attack_surface import WordSubstitutionAttackSurface, LMConstrainedAttackSurface
    if opt.lm_constraint:
        attack_surface = LMConstrainedAttackSurface.from_files(opt.certified_neighbors_file, opt.imdb_lm_file)
    else:
        attack_surface = WordSubstitutionAttackSurface.from_files(opt.certified_neighbors_file, opt.imdb_lm_file)
    
    if opt.resume != None:
        model = set_params(model, opt.resume, data_parallel=True, bert=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model.cuda()
        #model=torch.nn.DataParallel(model)

    from modified_bert_tokenizer import ModifiedBertTokenizer
    tokenizer = ModifiedBertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    acc=utils.evaluation_bert(opt, device, model, test_iter)
    print("test acc %.4f" % (acc))
    adv_acc=utils.evaluation_adv_bert(opt, device, model, test_iter, tokenizer)
    print("test g_adv_acc %.4f" % (adv_acc))
    genetic_attack(opt, device, model, attack_surface, dataset=opt.dataset, genetic_test_num=opt.genetic_test_num, test_bert=True)
    #fool_text_classifier_pytorch(opt, device, model,dataset=opt.dataset, clean_samples_cap=opt.pwws_test_num)
    
def main():
    opt = opts.parse_opt()
    print(opt)
    if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] =opt.gpu

    if opt.dataset == "imdb":
        syn_train_iter, syn_dev_iter, syn_test_iter, syn_data = make_synthesized_iter_for_bert(opt)

    if opt.out_syn_netx_file:
        file_name = "./syn_netx.txt"
        with open(file_name, "w") as f:
            for i, syn_list in enumerate(syn_data):
                node_x = i
                if len(syn_list) != 0:
                    for node_y in syn_list:
                        f.write(str(node_x)+" "+str(node_y)+"\n")

    test(opt, syn_test_iter, syn_data)
    
if __name__=="__main__": 
    main()