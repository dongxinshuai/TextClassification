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

def train(opt,train_iter, dev_iter, test_iter, syn_data, verbose=True):
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

    params = [param for param in model.parameters() if param.requires_grad] #filter(lambda p: p.requires_grad, model.parameters())
    #optimizer = utils.getOptimizer(params,name=opt.optimizer, lr=opt.learning_rate,weight_decay=opt.weight_decay,scheduler= utils.get_lr_scheduler(opt.lr_scheduler))
    from transformers import AdamW
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.learning_rate)
    
    scheduler = WarmupMultiStepLR(optimizer, (40, 80), 0.1, 1.0/10.0, 2, 'linear')

    from modified_bert_tokenizer import ModifiedBertTokenizer

    tokenizer = ModifiedBertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

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

    best_adv_acc = 0
    for epoch in range(11):

        if opt.smooth_ce:
            if epoch < 10:
                weight_adv = epoch*1.0/10
                weight_clean = 1-weight_adv
            else:
                weight_adv=1
                weight_clean=0
        else:
            weight_adv = opt.weight_adv
            weight_clean = opt.weight_clean

        if epoch>=opt.kl_start_epoch:
            kl_control = 1

        sum_loss = sum_loss_adv = sum_loss_kl = sum_loss_clean = 0
        total = 0

        asw = asw_count = sc=0


        for iters,batch in enumerate(train_iter):

            text = batch[0].to(device)
            label = batch[1].to(device)
            text_like_syn= batch[2].to(device)
            text_like_syn_valid= batch[3].to(device)
            bert_mask= batch[4].to(device)
            bert_token_id= batch[5].to(device)

            bs, sent_len = text.shape

            model.train()
            
            # zero grad
            optimizer.zero_grad()

            attack_type_dict = {
                'num_steps': opt.train_attack_iters,
                'loss_func': 'ce',
                'w_optm_lr': opt.w_optm_lr,
                'sparse_weight': opt.train_attack_sparse_weight,
                'out_type': "comb_p"
            }

            with torch.no_grad():
                embd = model(mode="text_to_embd", input=text, bert_mask=bert_mask, bert_token_id=bert_token_id) #in bs, len sent, vocab
            n,l,s = text_like_syn.shape
            with torch.no_grad():
                text_like_syn_embd = model(mode="text_to_embd", input=text_like_syn.reshape(n*l,s), bert_mask=bert_mask.reshape(n,l,1).repeat(1,1,s).reshape(n*l,s), bert_token_id=bert_token_id.reshape(n,l,1).repeat(1,1,s).reshape(n*l,s)).reshape(n,l,s,-1)
            adv_comb_p = model(mode="get_adv_by_convex_syn", input=embd, label=label, text_like_syn_embd=text_like_syn_embd, text_like_syn_valid=text_like_syn_valid, attack_type_dict=attack_type_dict, bert_mask=bert_mask, bert_token_id=bert_token_id)
            
            optimizer.zero_grad()
            # clean loss
            predicted = model(mode="text_to_logit", input=text, bert_mask=bert_mask, bert_token_id=bert_token_id)
            loss_clean= loss_fun(predicted,label)
            # adv loss

            predicted_adv = model(mode="text_syn_p_to_logit", input=text_like_syn, comb_p=adv_comb_p, bert_mask=bert_mask, bert_token_id=bert_token_id)

            loss_adv = loss_fun(predicted_adv,label)
            # kl loss
            criterion_kl = nn.KLDivLoss(reduction="sum")
            loss_kl = (1.0 / bs) * criterion_kl(F.log_softmax(predicted_adv, dim=1),
                                                        F.softmax(predicted, dim=1))

            # optimize
            loss =  opt.weight_kl * kl_control * loss_kl + weight_adv * loss_adv + weight_clean * loss_clean
            loss.backward() 
            optimizer.step()
            sum_loss += loss.item()
            sum_loss_adv += loss_adv.item()
            sum_loss_clean += loss_clean.item()
            sum_loss_kl += loss_kl.item()
            predicted, idx = torch.max(predicted, 1) 
            precision=(idx==label).float().mean().item()
            predicted_adv, idx = torch.max(predicted_adv, 1)
            precision_adv=(idx==label).float().mean().item()
            total += 1

            out_log = "%d epoch %d iters: loss: %.3f, loss_kl: %.3f, loss_adv: %.3f, loss_clean: %.3f | acc: %.3f acc_adv: %.3f | in %.3f seconds" % (epoch, iters, sum_loss/total, sum_loss_kl/total, sum_loss_adv/total, sum_loss_clean/total, precision, precision_adv, time.time()-start)
            start= time.time()
            logger.info(out_log)
            print(out_log)
                

        scheduler.step()

        if epoch%1==0:
            acc=utils.evaluation_bert(opt, device, model, dev_iter)
            out_log="%d epoch with dev acc %.4f" % (epoch,acc)
            logger.info(out_log)
            print(out_log)
            adv_acc=utils.evaluation_adv_bert(opt, device, model, dev_iter, tokenizer)
            out_log="%d epoch with dev g_adv_acc %.4f" % (epoch,adv_acc)
            logger.info(out_log)
            print(out_log)

            genetic_adv_acc=genetic_attack(opt, device, model, attack_surface, dataset=opt.dataset, genetic_test_num=20, test_bert=True)
            out_log="%d epoch with genetic adv acc %.4f" % (epoch, genetic_adv_acc)
            logger.info(out_log)
            print(out_log)

            #hotflip_adv_acc=utils.evaluation_hotflip_adv(opt, device, model, dev_iter, tokenizer)
            #out_log="%d epoch with dev hotflip adv acc %.4f" % (epoch,hotflip_adv_acc)
            #logger.info(out_log)
            #print(out_log)

            current_model_path=os.path.join(opt.out_path, "{}_epoch{}.pth".format(opt.model, epoch))
            state = {
                    'net': model.state_dict(),
                    'epoch': epoch,
                }
            torch.save(state, current_model_path)

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
    model = set_params(model, best_save_dir, data_parallel=True, bert=True)
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

    train(opt,syn_train_iter, syn_dev_iter, syn_test_iter, syn_data)
    
if __name__=="__main__": 
    main()