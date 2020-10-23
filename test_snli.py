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

from dataHelper_PWWS import load_synonyms_in_vocab, snli_make_synthesized_iter
from PWWS.fool_pytorch import fool_text_classifier_pytorch_snli, genetic_attack_snli

import opts
import models
import utils

from solver.lr_scheduler import WarmupMultiStepLR

timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time()) ))
performance_log_file =  os.path.join("log","result"+timeStamp+ ".csv") 
if not os.path.exists(performance_log_file):
    with open(performance_log_file,"w") as f:
        f.write("argument\n")
        f.close() 
      

def set_params(net, resume_model_path, data_parallel=False):
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(resume_model_path), 'Error: ' + resume_model_path + 'checkpoint not found!'
    checkpoint = torch.load(resume_model_path)
    state_dict = checkpoint['net']
    from collections import OrderedDict
    sdict = OrderedDict()
    for key in state_dict.keys():
        new_key = key.split('module.')[-1]
        if data_parallel:
            new_key = 'module.'+new_key
        sdict[new_key]=state_dict[key]
    net.load_state_dict(sdict)
    return net


def test(opt,train_iter, test_iter, verbose=True):
    global_start= time.time()
    logger = utils.getLogger()
    model=models.setup(opt)

    if opt.resume != None:
        model = set_params(model, opt.resume)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model.cuda()
        #model=torch.nn.DataParallel(model)

    from PWWS.word_level_process import word_process, get_tokenizer
    tokenizer = get_tokenizer(opt.dataset)
    #adv_acc=utils.snli_evaluation_adv(opt, device, model, test_iter, tokenizer)

    from from_certified.attack_surface import WordSubstitutionAttackSurface, LMConstrainedAttackSurface
    if opt.lm_constraint:
        attack_surface = LMConstrainedAttackSurface.from_files(opt.certified_neighbors_file, opt.snli_lm_file)
    else:
        attack_surface = WordSubstitutionAttackSurface.from_file(opt.certified_neighbors_file)

    genetic_attack_snli(opt, device, model, attack_surface, dataset=opt.dataset, genetic_test_num=opt.genetic_test_num)

    fool_text_classifier_pytorch_snli(opt, device, model,dataset=opt.dataset, clean_samples_cap=opt.pwws_test_num)

def main():
    parameter_pools = utils.parse_grid_parameters("config/grid_search_cnn.ini")
    
#    parameter_pools={
#            "model":["lstm","cnn","fasttext"],
#            "keep_dropout":[0.8,0.9,1.0],
#            "batch_size":[32,64,128],
#            "learning_rate":[100,10,1,1e-1,1e-2,1e-3],
#            "optimizer":["adam"],
#            "lr_scheduler":[None]            
#                        }    
    opt = opts.parse_opt()
    print(opt)
    if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] =opt.gpu

    syn_train_iter, syn_dev_iter, syn_test_iter, syn_data = snli_make_synthesized_iter(opt)
    test(opt,syn_train_iter, syn_test_iter)
    
   

if __name__=="__main__": 
    main()