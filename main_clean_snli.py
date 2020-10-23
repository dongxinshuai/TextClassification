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

from dataHelper_PWWS import load_synonyms_in_vocab, make_synthesized_iter, snli_make_synthesized_iter
from PWWS.fool_pytorch import fool_text_classifier_pytorch_snli, genetic_attack_snli

import opts
import models
import utils

from solver.lr_scheduler import WarmupMultiStepLR

try:
    import cPickle as pickle
except ImportError:
    import pickle

timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time()) ))
performance_log_file =  os.path.join("log","result"+timeStamp+ ".csv") 
if not os.path.exists(performance_log_file):
    with open(performance_log_file,"w") as f:
        f.write("argument\n")
        f.close() 
      

def set_params(net, resume_model_path, data_parallel=False, layer_keyword=None):
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
    if layer_keyword is None:
        net.load_state_dict(sdict)
    else:
        filtered_sdict = {}
        for key in state_dict.keys():
            if key in layer_keyword:
                filtered_sdict[key] = state_dict[key]
        net.load_state_dict(filtered_sdict, strict=False)
    return net


def train(opt,train_iter, dev_iter, test_iter, syn_data, verbose=True):
    global_start= time.time()
    logger = utils.getLogger()
    model=models.setup(opt)

    from from_certified.attack_surface import WordSubstitutionAttackSurface, LMConstrainedAttackSurface
    if opt.lm_constraint:
        attack_surface = LMConstrainedAttackSurface.from_files(opt.certified_neighbors_file, opt.snli_lm_file)
    else:
        attack_surface = WordSubstitutionAttackSurface.from_file(opt.certified_neighbors_file)

    if opt.resume != None:
        if opt.resume_vector_only:
            model = set_params(model, opt.resume, layer_keyword=['embedding.weight', 'linear_transform_embd_1.weight' ])
        else:
            model = set_params(model, opt.resume)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model.cuda()
        #model=torch.nn.DataParallel(model)

    # set optimizer
    if opt.embd_freeze == True:
        model.embedding.weight.requires_grad = False
    else:
        model.embedding.weight.requires_grad = True

    if opt.embd_transform:
        if opt.embd_fc_freeze == True:
            model.linear_transform_embd_1.weight.requires_grad = False
        else:
            model.linear_transform_embd_1.weight.requires_grad = True

    params = [param for param in model.parameters() if param.requires_grad] #filter(lambda p: p.requires_grad, model.parameters())
    optimizer = utils.getOptimizer(params,name=opt.optimizer, lr=opt.learning_rate,weight_decay=opt.weight_decay,scheduler= utils.get_lr_scheduler(opt.lr_scheduler))
    scheduler = WarmupMultiStepLR(optimizer, (40, 70), 0.1, 1.0/10.0, 2, 'linear')

    from label_smooth import LabelSmoothSoftmaxCE
    if opt.label_smooth!=0:
        assert(opt.label_smooth<=1 and opt.label_smooth>0)
        loss_fun = LabelSmoothSoftmaxCE(lb_pos=1-opt.label_smooth, lb_neg=opt.label_smooth)
    else:
        loss_fun = F.cross_entropy

    filename = None
    acc_adv_list=[]
    start= time.time()
    kl_control = 0

    # initialize synonyms with the same embd
    from PWWS.word_level_process import word_process, get_tokenizer
    tokenizer = get_tokenizer(opt.dataset)

    #g_adv_acc=utils.snli_evaluation_adv(opt, device, model,test_iter, tokenizer)
    #adv_acc = genetic_attack_snli(opt, device, model, attack_surface, dataset=opt.dataset, genetic_test_num=opt.genetic_test_num)

    best_adv_acc = 0
    for epoch in range(31):

        if epoch>=opt.kl_start_epoch:
            kl_control = 1

        sum_loss = sum_loss_adv = sum_loss_kl = sum_loss_clean = 0
        total = 0

        for iters,batch in enumerate(train_iter):
            x_p = batch[0].to(device)
            x_h = batch[1].to(device)
            label = batch[2].to(device)
            x_p_text_like_syn= batch[3].to(device)
            x_p_text_like_syn_valid= batch[4].to(device)
            x_h_text_like_syn= batch[5].to(device)
            x_h_text_like_syn_valid= batch[6].to(device)
            x_p_mask= batch[7].to(device)
            x_h_mask= batch[8].to(device)

            bs, sent_len = x_p.shape

            model.train()
            
            # zero grad
            optimizer.zero_grad()
            # clean loss
            #predicted = model(mode="embd_to_logit", input=embd)
            predicted = model(mode="text_to_logit", x_p=x_p, x_h=x_h ,x_p_mask=x_p_mask, x_h_mask=x_h_mask,)
            loss_clean= loss_fun(predicted,label)

            # optimize
            loss = loss_clean
            loss.backward() 
            optimizer.step()
            sum_loss += loss.item()
            sum_loss_clean += loss_clean.item()
            predicted, idx = torch.max(predicted, 1) 
            precision=(idx==label).float().mean().item()
            total += 1

            out_log = "%d epoch %d iters: loss: %.3f, loss_clean: %.3f | acc: %.3f | in %.3f seconds" % (epoch, iters, sum_loss/total, sum_loss_clean/total, precision, time.time()-start)
            start= time.time()
            logger.info(out_log)
            print(out_log)
                
        scheduler.step()
        
        if epoch%2==0:
            acc=utils.snli_evaluation(opt, device, model, dev_iter)
            out_log="%d epoch with dev acc %.4f" % (epoch,acc)
            logger.info(out_log)
            print(out_log)

            adv_acc=utils.snli_evaluation_adv(opt, device, model, dev_iter, tokenizer)
            out_log="%d epoch with dev g_adv_acc %.4f" % (epoch,adv_acc)
            logger.info(out_log)
            print(out_log)
            
            #fool_text_classifier_pytorch_snli(opt, device, model,dataset=opt.dataset, clean_samples_cap=opt.pwws_test_num)
            #adv_acc = genetic_attack_snli(opt, device, model, attack_surface, dataset=opt.dataset, genetic_test_num=opt.genetic_test_num)

            if adv_acc>=best_adv_acc:
                best_adv_acc = adv_acc
                best_save_dir=os.path.join(opt.out_path, "{}_best.pth".format(opt.model))
                #model_path=os.path.join(opt.out_path, "{}_epoch{}.pth".format(opt.model, epoch))
                state = {
                    'net': model.state_dict(),
                    'epoch': epoch,
                }
                torch.save(state, best_save_dir)

    # restore best according to dev set
    model = set_params(model, best_save_dir)
    acc=utils.snli_evaluation(opt, device, model, test_iter)
    print("test acc %.4f" % (acc))
    adv_acc=utils.snli_evaluation_adv(opt, device, model, test_iter, tokenizer)
    print("test g_adv_acc %.4f" % (adv_acc))

    genetic_attack_snli(opt, device, model, attack_surface, dataset=opt.dataset, genetic_test_num=opt.genetic_test_num)
    fool_text_classifier_pytorch_snli(opt, device, model,dataset=opt.dataset, clean_samples_cap=opt.pwws_test_num)

        
def main():
    opt = opts.parse_opt()
    print(opt)
    assert(opt.dataset=="snli")
    if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] =opt.gpu
    if opt.dataset == "snli":
        syn_train_iter, syn_dev_iter, syn_test_iter, syn_data = snli_make_synthesized_iter(opt)

    if opt.out_syn_netx_file:
        file_name = "./syn_netx.txt"
        with open(file_name, "w") as f:
            for i, syn_list in enumerate(syn_data):
                node_x = i
                if len(syn_list) != 0:
                    for node_y in syn_list:
                        f.write(str(node_x)+" "+str(node_y)+"\n")

    train(opt,syn_train_iter, syn_dev_iter, syn_test_iter, syn_data)
    
if __name__=="__main__": 
    main()