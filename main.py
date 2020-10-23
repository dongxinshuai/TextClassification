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

from dataHelper_PWWS import load_synonyms_in_vocab, make_synthesized_iter
from PWWS.fool_pytorch import fool_text_classifier_pytorch

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


def train(opt,train_iter, test_iter, syn_data, verbose=True):
    global_start= time.time()
    logger = utils.getLogger()
    model=models.setup(opt)

    if opt.resume != None:
        model = set_params(model, opt.resume)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model.cuda()
        #model=torch.nn.DataParallel(model)

    # set optimizer
    model.embedding.weight.requires_grad = False
    params = [param for param in model.parameters() if param.requires_grad] #filter(lambda p: p.requires_grad, model.parameters())
    optimizer = utils.getOptimizer(params,name=opt.optimizer, lr=opt.learning_rate,weight_decay=opt.weight_decay,scheduler= utils.get_lr_scheduler(opt.lr_scheduler))
    scheduler = WarmupMultiStepLR(optimizer, (40, 80), 0.1, 1.0/10.0, 5, 'linear')

    model.embedding.weight.requires_grad = True
    #ball_params =  [param for param in model.parameters() if param.requires_grad]
    ball_params = [model.embedding.weight]
    ball_optimizer = utils.getOptimizer(ball_params,name=opt.ball_optimizer, lr=opt.ball_learning_rate,weight_decay=opt.ball_weight_decay,scheduler=None)
    ball_scheduler = WarmupMultiStepLR(ball_optimizer, (40, 80), 0.1, 1.0/10.0, 5, 'linear')
    #

    loss_fun = F.cross_entropy
    filename = None
    acc_adv_list=[]
    start= time.time()
    kl_control = 0

    # initialize synonyms with the same embd
    from PWWS.word_level_process import word_process, get_tokenizer
    tokenizer = get_tokenizer(opt.dataset)

    father_dict= {}
    for index in range(1+len(tokenizer.index_word)):
        father_dict[index] = index

    def get_father(x):
        if father_dict[x] == x:
            return x
        else:
            fa = get_father(father_dict[x])
            father_dict[x] = fa
            return fa

    for index in range(len(syn_data)-1, 0, -1):
        syn_list = syn_data[index]
        for pos in syn_list:
            fa_pos = get_father(pos)
            fa_anch = get_father(index)
            if fa_pos == fa_anch:
                father_dict[index] = index
                father_dict[fa_anch] = index
            else:
                father_dict[index] = index
                father_dict[fa_anch] = index
                father_dict[fa_pos] = index

    if opt.embedding_prep == "same":
        print("Same embedding for synonyms as embd prep.")
        set_different_embd=set()
        for key in father_dict:
            fa = get_father(key)
            set_different_embd.add(fa)
            with torch.no_grad():
                model.embedding.weight[key, :] = model.embedding.weight[fa, :]
        print(len(set_different_embd))

    elif opt.embedding_prep == "ge":
        print("Graph embedding as embd prep.")
        ge_file = opt.ge_file
        f=open(ge_file,'rb')
        saved=pickle.load(f)
        ge_embeddings_dict = saved['walk_embeddings']
        #model = saved['model']
        f.close()
        with torch.no_grad():
            for key in ge_embeddings_dict:
                model.embedding.weight[int(key), :] = torch.FloatTensor(ge_embeddings_dict[key])
    else:
        print("No embd prep.")

    #
    #fool_text_classifier_pytorch(opt, device, model,dataset=opt.dataset)

    for epoch in range(101):

        if epoch>=opt.kl_start_epoch:
            kl_control = 1

        for iters,batch in enumerate(train_iter):
            text = batch[0].to(device)
            label = batch[1].to(device)
            anch = batch[2].to(device)
            pos = batch[3].to(device)
            neg = batch[4].to(device)
            anch_valid= batch[5].to(device).unsqueeze(2)
            text_like_syn= batch[6].to(device)
            text_like_syn_valid= batch[7].to(device)

            bs, sent_len = text.shape

            model.train()
            
            # zero grad
            ball_optimizer.zero_grad()
            optimizer.zero_grad()

            if opt.pert_set=="l2_ball":
                set_radius = model(mode="text_to_radius", input=text) #in bs, len sent, 1
                #set_radius = opt.train_attack_eps
                attack_type_dict = {
                    'num_steps': opt.train_attack_iters,
                    'step_size': opt.train_attack_step_size * set_radius,
                    'random_start': opt.random_start,
                    'epsilon':  set_radius,
                    #'loss_func': 'ce',
                    'loss_func': 'kl',
                    'direction': 'away',
                }
                embd = model(mode="text_to_embd", input=text) #in bs, len sent, vocab
                embd_adv = model(mode="get_embd_adv", input=embd, label=label, attack_type_dict=attack_type_dict)
            elif opt.pert_set=="convex_combination":
                attack_type_dict = {
                    'num_steps': opt.train_attack_iters,
                    'loss_func': 'kl',
                    'w_optm_lr': opt.w_optm_lr,
                    'sparse_weight': opt.attack_sparse_weight,
                    'out_type': "embd"
                }
                embd = model(mode="text_to_embd", input=text) #in bs, len sent, vocab
                n,l,s = text_like_syn.shape
                text_like_syn_embd = model(mode="text_to_embd", input=text_like_syn.reshape(n,l*s)).reshape(n,l,s,-1)
                embd_adv = model(mode="get_adv_by_convex_syn", input=embd, label=label, text_like_syn_embd=text_like_syn_embd, text_like_syn_valid=text_like_syn_valid, attack_type_dict=attack_type_dict)
            elif opt.pert_set=="ad_text":
                attack_type_dict = {
                    'num_steps': opt.train_attack_iters,
                    'loss_func': 'ce',
                    'w_optm_lr': opt.w_optm_lr,
                    'sparse_weight': opt.attack_sparse_weight,
                    'out_type': "text"
                }
                embd = model(mode="text_to_embd", input=text) #in bs, len sent, vocab
                n,l,s = text_like_syn.shape
                text_like_syn_embd = model(mode="text_to_embd", input=text_like_syn.reshape(n,l*s)).reshape(n,l,s,-1)
                text_adv = model(mode="get_adv_by_convex_syn", input=embd, label=label, text_like_syn_embd=text_like_syn_embd, text_like_syn_valid=text_like_syn_valid, text_like_syn=text_like_syn, attack_type_dict=attack_type_dict)
            elif opt.pert_set=="ad_text_syn_p":
                attack_type_dict = {
                    'num_steps': opt.train_attack_iters,
                    'loss_func': 'ce',
                    'w_optm_lr': opt.w_optm_lr,
                    'sparse_weight': 0,
                    'out_type': "comb_p"
                }
                embd = model(mode="text_to_embd", input=text) #in bs, len sent, vocab
                n,l,s = text_like_syn.shape
                text_like_syn_embd = model(mode="text_to_embd", input=text_like_syn.reshape(n,l*s)).reshape(n,l,s,-1)
                adv_comb_p = model(mode="get_adv_by_convex_syn", input=embd, label=label, text_like_syn_embd=text_like_syn_embd, text_like_syn_valid=text_like_syn_valid, attack_type_dict=attack_type_dict)


            ball_optimizer.zero_grad()
            optimizer.zero_grad()
            # clean loss
            #predicted = model(mode="embd_to_logit", input=embd)
            predicted = model(mode="text_to_logit", input=text)
            loss_clean= loss_fun(predicted,label)
            # adv loss
            if opt.pert_set=="ad_text":
                predicted_adv = model(mode="text_to_logit", input=text_adv)
            elif opt.pert_set=="ad_text_syn_p":
                predicted_adv = model(mode="text_syn_p_to_logit", input=text_like_syn, comb_p=adv_comb_p)
            else:
                predicted_adv = model(mode="embd_to_logit", input=embd_adv)
            loss_adv = loss_fun(predicted_adv,label)
            # kl loss
            criterion_kl = nn.KLDivLoss(reduction="sum")
            loss_kl = (1.0 / bs) * criterion_kl(F.log_softmax(predicted_adv, dim=1),
                                                        F.softmax(predicted, dim=1))
            # l2 ball loss
            embd_anch = model(mode="text_to_embd", input=anch).unsqueeze(2) #n, anch len, 1, embed_dim
            embd_pos = model(mode="text_to_embd", input=pos) #n, anch len, pos_len, embed_dim
            embd_neg = model(mode="text_to_embd", input=neg) #n, anch len, neg_len, embed_dim

            batch_size, num_anch, _, embd_dim = embd_anch.shape

            ap = (embd_anch-embd_pos)
            _, _, num_pos, _ = ap.shape
            ap = ap.reshape(-1, embd_dim)
            ap_d = F.pairwise_distance(ap, torch.zeros_like(ap), p=2.0) #n * anch len * pos_len

            an = (embd_anch.detach() -embd_neg)
            _, _, num_neg, _ = an.shape
            an = an.reshape(-1, embd_dim)
            an_d = F.pairwise_distance(an, torch.zeros_like(an), p=2.0) #n*  anch len* neg_len

            #loss_an = F.relu(1-an_d).reshape(batch_size, num_anch, num_neg) * anch_valid#.mean() 
            #loss_ap = F.relu(ap_d-0.8).reshape(batch_size, num_anch, num_pos) * anch_valid#.mean()
            loss_ap = ap_d.reshape(batch_size, num_anch, num_pos) * anch_valid#.mean() 

            #loss_an=loss_an.sum()/(anch_valid.sum())/num_neg
            loss_ap=loss_ap.sum()/(anch_valid.sum())/num_pos
            #
            # optimize
            loss =  opt.weight_kl * kl_control * loss_kl + opt.weight_adv * loss_adv + opt.weight_clean * loss_clean + opt.weight_ball * (loss_ap)
            #loss =  opt.weight_kl * kl_control * loss_kl + opt.weight_adv * loss_adv + opt.weight_clean * loss_clean + opt.weight_ball * (loss_ap)
            loss.backward() 
            optimizer.step()
            ball_optimizer.step()

            out_log = "%d epoch %d iters E-step: with loss: %.5f, loss_kl: %.5f, loss_adv: %.5f, loss_clean: %.5f loss_ap: %.5f in %.4f seconds" % (epoch,iters,loss.cpu().data.numpy(),loss_kl.cpu().data.numpy(),loss_adv.cpu().data.numpy(),loss_clean.cpu().data.numpy(),loss_ap.cpu().data.numpy(),time.time()-start)
            start= time.time()
            if verbose:
                logger.info(out_log)
                print(out_log)
                
########### update radius
            with torch.no_grad():
                embd_anch = model(mode="text_to_embd", input=anch).unsqueeze(2) #n, anch len, 1, embed_dim
                embd_pos = model(mode="text_to_embd", input=pos) #n, anch len, pos_len, embed_dim

                batch_size, num_anch, _, embd_dim = embd_anch.shape

                ap = (embd_anch-embd_pos)
                _, _, num_pos, _ = ap.shape
                ap = ap.reshape(-1, embd_dim)
                ap_d = F.pairwise_distance(ap, torch.zeros_like(ap), p=2.0) #n * anch len * pos_len

                get_radius_max, _ = (ap_d.reshape(batch_size, num_anch, num_pos)).max(-1)
                get_radius_mean    = (ap_d.reshape(batch_size, num_anch, num_pos)).mean(-1)
                loss_ap_max = 0
                for i in range(batch_size):
                    for j in range(num_anch):
                        if int(anch_valid[i,j,0].cpu().numpy())!=0:
                            anch_word = anch[i,j]
                            model.word_synonym_radius.weight[anch_word, 0] = get_radius_mean[i,j]

                            loss_ap_max += get_radius_max[i,j]

                loss_ap_max = loss_ap_max/(anch_valid.sum())

            out_log = "loss_ap_max: %.5f" % (loss_ap_max.cpu().data.numpy())
            if verbose:
                logger.info(out_log)
                print(out_log)

        scheduler.step()
        ball_scheduler.step()

        if epoch%5==0:
            acc=utils.evaluation(opt, device, model,test_iter,opt.from_torchtext)
            if verbose:
                out_log="%d epoch with acc %.4f" % (epoch,acc)
                logger.info(out_log)
                print(out_log)

            acc_adv=utils.evaluation_adv(opt, device, model,test_iter,opt.from_torchtext)
            if verbose:
                out_log="%d epoch with acc_adv %.4f" % (epoch,acc_adv)
                logger.info(out_log)
                print(out_log)

        if epoch%5==0 and epoch!=0:
            fool_text_classifier_pytorch(opt, device, model,dataset=opt.dataset)

        if epoch%20==0 and epoch!=0:
            model_path=os.path.join(opt.out_path, "{}_epoch{}.pth".format(opt.model, epoch))
            #model_path=os.path.join(opt.out_path, "{}_best.pth".format(opt.model))
            state = {
                'net': model.state_dict(),
                'acc': acc_adv, 
                'epoch': epoch,
            }
            torch.save(state, model_path)

        acc_adv_list.append(acc_adv)


        
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
    #train_iter, test_iter = utils.loadData(opt)
    #synonym_train_iter = load_synonyms_in_vocab(opt)
    syn_train_iter, syn_dev_iter, syn_test_iter, syn_data = make_synthesized_iter(opt)

#    if from_torchtext:
#        train_iter, test_iter = utils.loadData(opt)
#    else:
#        import dataHelper 
#        train_iter, test_iter = dataHelper.loadData(opt)

    if opt.out_syn_netx_file:
        file_name = "./syn_netx.txt"
        with open(file_name, "w") as f:
            for i, syn_list in enumerate(syn_data):
                node_x = i
                if len(syn_list) != 0:
                    for node_y in syn_list:
                        f.write(str(node_x)+" "+str(node_y)+"\n")

    train(opt,syn_train_iter, syn_test_iter, syn_data)
    
    # if False:
    #     model=models.setup(opt)
    #     print(opt.model)
    #     if torch.cuda.is_available():
    #         model.cuda()
    #     train(opt,train_iter, test_iter)
    # else:
        
    #     pool =[ arg for arg in itertools.product(*parameter_pools.values())]
    #     random.shuffle(pool)
    #     args=[arg for i,arg in enumerate(pool) if i%opt.gpu_num==opt.gpu]
        
    #     for arg in args:
    #         olddataset = opt.dataset
    #         for k,v in zip(parameter_pools.keys(),arg):
    #             opt.__setattr__(k,v)
    #         if "dataset" in parameter_pools and olddataset != opt.dataset:
    #             train_iter, test_iter = utils.loadData(opt)
    #         train(opt,train_iter, test_iter,verbose=False)
   

if __name__=="__main__": 
    main()