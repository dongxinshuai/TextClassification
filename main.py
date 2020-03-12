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

timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time()) ))
performance_log_file =  os.path.join("log","result"+timeStamp+ ".csv") 
if not os.path.exists(performance_log_file):
    with open(performance_log_file,"w") as f:
        f.write("argument\n")
        f.close() 
      
        
def train(opt,train_iter, test_iter, verbose=True):
    global_start= time.time()
    logger = utils.getLogger()
    model=models.setup(opt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model.cuda()
        #model=torch.nn.DataParallel(model)
    """
    model_info =";".join( [str(k)+":"+ str(v)  for k,v in opt.__dict__.items() if type(v) in (str,int,float,list,bool)])  
    logger.info("# parameters:" + str(sum(param.numel() for param in params)))
    logger.info(model_info)
    """

    model.embedding.weight.requires_grad = False
    params = [param for param in model.parameters() if param.requires_grad] #filter(lambda p: p.requires_grad, model.parameters())
    optimizer = utils.getOptimizer(params,name=opt.optimizer, lr=opt.learning_rate,weight_decay=opt.weight_decay,scheduler= utils.get_lr_scheduler(opt.lr_scheduler))
    scheduler = WarmupMultiStepLR(optimizer, (40, 80), 0.1, 1.0/10.0, 5, 'linear')

    model.embedding.weight.requires_grad = True
    #ball_params =  [param for param in model.parameters() if param.requires_grad]
    ball_params = [model.embedding.weight]
    ball_optimizer = utils.getOptimizer(ball_params,name=opt.ball_optimizer, lr=opt.ball_learning_rate,weight_decay=opt.ball_weight_decay,scheduler=None)
    ball_scheduler = WarmupMultiStepLR(ball_optimizer, (40, 80), 0.1, 1.0/10.0, 5, 'linear')

    loss_fun = F.cross_entropy

    filename = None
    acc_adv_list=[]
    start= time.time()
    
    """
    iter_synsynonym_train_iter = synonym_train_iter.__iter__()
    try:
        batch_synonyms = next(iter_synsynonym_train_iter)
    except:
        iter_synsynonym_train_iter = synonym_train_iter.__iter__()
        batch_synonyms = next(iter_synsynonym_train_iter)
    roots = batch_synonyms[0].to(device)
    synonyms = batch_synonyms[1].to(device)
    negatives = batch_synonyms[2].to(device)
    """
    
    for epoch in range(100):
        for iters,batch in enumerate(train_iter):
        #for iters, batch_synonyms in enumerate(synonym_train_iter):
            text = batch[0].to(device)
            label = batch[1].to(device)
            anch = batch[2].to(device)
            pos = batch[3].to(device)
            neg = batch[4].to(device)
            anch_valid= batch[5].to(device).unsqueeze(2)


            bs, sent_len = text.shape

            model.train()

            set_radius = opt.train_attack_eps
            attack_type_dict = {
                'num_steps': opt.train_attack_iters,
                'step_size': opt.train_attack_step_size * set_radius,
                'random_start': opt.random_start,
                'epsilon':  set_radius,
                'loss_func': 'ce',
                'direction': 'away',
            }

            embd = model(mode="text_to_embd", input=text) #in bs, len sent, vocab
            embd_adv = model(mode="get_embd_adv", input=embd, label=label, attack_type_dict=attack_type_dict)
            # zero grad
            ball_optimizer.zero_grad()
            optimizer.zero_grad()
            # l2 ball
            embd_anch = model(mode="text_to_embd", input=anch).unsqueeze(2) #n, anch len, 1, embed_dim
            embd_pos = model(mode="text_to_embd", input=pos) #n, anch len, pos_len, embed_dim
            embd_neg = model(mode="text_to_embd", input=neg) #n, anch len, neg_len, embed_dim

            batch_size, num_anch, _, embd_dim = embd_anch.shape

            ap = (embd_anch-embd_pos)
            _, _, num_pos, _ = ap.shape
            ap = ap.reshape(-1, embd_dim)
            ap_d = F.pairwise_distance(ap, torch.zeros_like(ap), p=2.0) #n * anch len * pos_len
            #ap_d=ap_d.reshape(batch_size, num_anch, num_pos)

            an = (embd_anch -embd_neg)
            _, _, num_neg, _ = an.shape
            an = an.reshape(-1, embd_dim)
            an_d = F.pairwise_distance(an, torch.zeros_like(an), p=2.0) #n*  anch len* neg_len

            loss_an = F.relu(set_radius-an_d).reshape(batch_size, num_anch, num_neg) * anch_valid#.mean() 
            loss_ap = F.relu(ap_d-set_radius).reshape(batch_size, num_anch, num_pos) * anch_valid#.mean() 

            #loss_ap = ap_d.mean() 
            loss_ap_max = loss_ap.max()

            loss_an=loss_an.mean()
            loss_ap=loss_ap.mean()

            loss_ball = (loss_an+loss_ap)
            #loss_ball = loss_ap
            # clean
            predicted = model(mode="text_to_logit",input=text)
            loss_clean= loss_fun(predicted,label)
            # adv
            predicted_adv = model(mode="embd_to_logit", input=embd_adv)
            loss_adv = loss_fun(predicted_adv,label)
            # optimize
            loss = opt.weight_adv * loss_adv + opt.weight_clean * loss_clean + opt.weight_ball * loss_ball
            loss.backward()
            #utils.clip_gradient(optimizer, opt.grad_clip)
            utils.sign_scale_gradient(optimizer, 0.001)
            utils.sign_scale_gradient(ball_optimizer, 0.01)
            ball_optimizer.step()
            optimizer.step()

            out_log = "%d epoch %d iters with loss: %.5f loss_adv: %.5f, loss_clean: %.5f loss_ap: %.5f loss_ap_max: %.5f loss_an: %.5f in %.4f seconds" % (epoch,iters,loss.cpu().data.numpy(),loss_adv.cpu().data.numpy(),loss_clean.cpu().data.numpy(),loss_ap.cpu().data.numpy(),loss_ap_max.cpu().data.numpy(),loss_an.cpu().data.numpy(),time.time()-start)
            start= time.time()
            if verbose:
                logger.info(out_log)
                print(out_log)

        if epoch%5==0:
            acc=utils.evaluation(opt, model,test_iter,opt.from_torchtext)
            if verbose:
                out_log="%d epoch with acc %.4f" % (epoch,acc)
                logger.info(out_log)
                print(out_log)

            acc_adv=utils.evaluation_adv(opt, model,test_iter,opt.from_torchtext)
            if verbose:
                out_log="%d epoch with acc_adv %.4f" % (epoch,acc_adv)
                logger.info(out_log)
                print(out_log)
        if epoch%20==0 and epoch!=0:
            fool_text_classifier_pytorch(device, model,dataset=opt.dataset)

        if len(acc_adv_list)==0 or acc_adv > max(acc_adv_list):
            model_path=os.path.join(opt.out_path, "{}_epoch{}.pth".format(opt.model, epoch))
            state = {
                'net': model.state_dict(),
                'acc': acc_adv, 
                'epoch': epoch,
            }
            torch.save(state, model_path)

        acc_adv_list.append(acc_adv)

    """
#    while(utils.is_writeable(performance_log_file)):
    df = pd.read_csv(performance_log_file,index_col=0,sep="\t")
    df.loc[model_info,opt.dataset] =  max(percisions) 
    df.to_csv(performance_log_file,sep="\t")    
    logger.info(model_info +" with time :"+ str( time.time()-global_start)+" ->" +str( max(percisions) ) )
    print(model_info +" with time :"+ str( time.time()-global_start)+" ->" +str( max(percisions) ) )
    """
        
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
    train_iter, test_iter = utils.loadData(opt)
    #synonym_train_iter = load_synonyms_in_vocab(opt)
    syn_train_iter, syn_test_iter = make_synthesized_iter(opt)

#    if from_torchtext:
#        train_iter, test_iter = utils.loadData(opt)
#    else:
#        import dataHelper 
#        train_iter, test_iter = dataHelper.loadData(opt)

    train(opt,syn_train_iter, test_iter)
    
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