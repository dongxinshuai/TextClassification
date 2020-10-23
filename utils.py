# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
import numpy as np
from functools import wraps
import time
import sys
import logging
import os,configparser,re
try:
    import cPickle as pickle
except ImportError:
    import pickle

def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco  

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:       
            if param.grad is not None and param.requires_grad:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def sign_scale_gradient(optimizer, scale):
    for group in optimizer.param_groups:
        for param in group['params']:       
            if param.grad is not None and param.requires_grad:
                param.grad.data =  torch.sign(param.grad.data)*scale

def loadData(opt):
    if not opt.from_torchtext:
        if opt.from_PWWS:
            import dataHelper_PWWS as helper
        else:
            import dataHelper as helper

        return helper.loadData(opt)
    else:
        device = 0 if  torch.cuda.is_available()  else -1

        TEXT = data.Field(lower=True, include_lengths=True, batch_first=True,fix_length=opt.max_seq_len)
        LABEL = data.Field(sequential=False)
        if opt.dataset=="imdb":
            train, test = datasets.IMDB.splits(TEXT, LABEL)
        elif opt.dataset=="sst":
            train, val, test = datasets.SST.splits( TEXT, LABEL, fine_grained=True, train_subtrees=True,
                                                filter_pred=lambda ex: ex.label != 'neutral')
        elif opt.dataset=="trec":
            train, test = datasets.TREC.splits(TEXT, LABEL, fine_grained=True)
        else:
            print("does not support this datset")
            
        TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
        LABEL.build_vocab(train)    
        # print vocab information
        print('len(TEXT.vocab)', len(TEXT.vocab))
        print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

        train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=opt.batch_size,device=device,repeat=False,shuffle=True)

        opt.label_size= len(LABEL.vocab)    
        opt.vocab_size = len(TEXT.vocab)
        opt.embedding_dim= TEXT.vocab.vectors.size()[1]
        opt.embeddings = TEXT.vocab.vectors
        
        return train_iter, test_iter

def snli_evaluation(opt, device, model,test_iter):
    model.eval()
    accuracy=[]
#    batch= next(iter(test_iter))
    for index,batch in enumerate( test_iter):
        x_p = batch[0].to(device)
        x_h = batch[1].to(device)
        label = batch[2].to(device)
        x_p_mask= batch[7].to(device)
        x_h_mask= batch[8].to(device)

        predicted = model(mode='text_to_logit',x_p=x_p, x_h=x_h,x_p_mask=x_p_mask, x_h_mask=x_h_mask)
        prob, idx = torch.max(predicted, 1) 
        percision=(idx==label).float().mean()
        
        if torch.cuda.is_available():
            accuracy.append(percision.data.item() )
        else:
            accuracy.append(percision.data.numpy()[0] )
    model.train()
    return np.mean(accuracy)

def evaluation(opt, device, model,test_iter):
    model.eval()
    accuracy=[]
#    batch= next(iter(test_iter))
    for index,batch in enumerate( test_iter):
        text = batch[0].to(device)
        label = batch[1].to(device)

        predicted = model(mode='text_to_logit',input=text)
        prob, idx = torch.max(predicted, 1) 
        percision=(idx==label).float().mean()
        
        if torch.cuda.is_available():
            accuracy.append(percision.data.item() )
        else:
            accuracy.append(percision.data.numpy()[0] )
    model.train()
    return np.mean(accuracy)


def snli_evaluation_adv(opt, device, model,test_iter,tokenizer):
    model.eval()
    accuracy=[]
#    batch= next(iter(test_iter))
    for index,batch in enumerate( test_iter):
        x_p = batch[0].to(device)
        x_h = batch[1].to(device)
        label = batch[2].to(device)
        x_p_text_like_syn= batch[3].to(device)
        x_p_text_like_syn_valid= batch[4].to(device)
        x_h_text_like_syn= batch[5].to(device)
        x_h_text_like_syn_valid= batch[6].to(device)
        x_p_mask= batch[7].to(device)
        x_h_mask= batch[8].to(device)

        batch_size = len(x_p)
        if index*batch_size > 9842:
            break

        attack_type_dict = {
            'num_steps': opt.test_attack_iters,
            'loss_func': 'ce',
            'w_optm_lr': opt.w_optm_lr,
            'sparse_weight': opt.attack_sparse_weight,
            'out_type': "text",
            'attack_hypo_only': True,
        }
        embd_p, embd_h = model(mode="text_to_embd", x_p=x_p, x_h=x_h) #in bs, len sent, vocab
        assert(x_p_text_like_syn.shape == x_h_text_like_syn.shape)
        n,l,s = x_p_text_like_syn.shape
        x_p_text_like_syn_embd, x_h_text_like_syn_embd = model(mode="text_to_embd", x_p=x_p_text_like_syn.reshape(n,l*s), x_h=x_h_text_like_syn.reshape(n,l*s))
        x_p_text_like_syn_embd = x_p_text_like_syn_embd.reshape(n,l,s,-1)
        x_h_text_like_syn_embd = x_h_text_like_syn_embd.reshape(n,l,s,-1)

        adv_x_p, adv_x_h = model(mode="get_adv_by_convex_syn", x_p=embd_p, x_h=embd_h, label=label, 
            x_p_text_like_syn = x_p_text_like_syn,
            x_p_text_like_syn_embd=x_p_text_like_syn_embd, x_p_text_like_syn_valid=x_p_text_like_syn_valid, 
            x_h_text_like_syn = x_h_text_like_syn,
            x_h_text_like_syn_embd=x_h_text_like_syn_embd, x_h_text_like_syn_valid=x_h_text_like_syn_valid,
            x_p_mask=x_p_mask, x_h_mask=x_h_mask,
            attack_type_dict=attack_type_dict)

        predicted = model(mode='text_to_logit',x_p=x_p, x_h=adv_x_h, x_p_mask=x_p_mask, x_h_mask=x_h_mask)
        #print("_________________________________")
        #print(inverse_tokenize(tokenizer, x_h[0]))
        #print(inverse_tokenize(tokenizer, adv_x_h[0]))

        prob, idx = torch.max(predicted, 1) 
        percision=(idx==label).float().mean()
        
        if torch.cuda.is_available():
            accuracy.append(percision.data.item() )
        else:
            accuracy.append(percision.data.numpy()[0] )
    model.train()
    return np.mean(accuracy)

def evaluation_adv(opt, device, model, test_iter, tokenizer):
    model.eval()
    accuracy=[]
    record_for_vis = {}
    record_for_vis["comb_p_list"] = []
    record_for_vis["embd_syn_list"] = []
    record_for_vis["syn_valid_list"] = []
    record_for_vis["text_syn_list"] = []

#    batch= next(iter(test_iter))
    if opt.pert_set=="convex_combination":
        print("ad test by convex_combination.")
    for index,batch in enumerate( test_iter):
        text = batch[0].to(device)
        label = batch[1].to(device)
        text_like_syn= batch[6].to(device)
        text_like_syn_valid= batch[7].to(device)

        batch_size = len(text)
        #if index*batch_size > 1000:
        #    break

        attack_type_dict = {
            'num_steps': opt.test_attack_iters,
            'loss_func': 'ce',
            'w_optm_lr': opt.w_optm_lr,
            'sparse_weight': opt.attack_sparse_weight,
            'out_type': "text"
        }
        embd = model(mode="text_to_embd", input=text) #in bs, len sent, vocab
        n,l,s = text_like_syn.shape
        text_like_syn_embd = model(mode="text_to_embd", input=text_like_syn.reshape(n,l*s)).reshape(n,l,s,-1)
        text_adv = model(mode="get_adv_by_convex_syn", input=embd, label=label, text_like_syn_embd=text_like_syn_embd, text_like_syn_valid=text_like_syn_valid, text_like_syn=text_like_syn, attack_type_dict=attack_type_dict, text_for_vis=text, record_for_vis=record_for_vis)
        predicted_adv = model(mode="text_to_logit", input=text_adv)

        #print("_________________________________")
        #print(inverse_tokenize(tokenizer, text[0]))
        #print(inverse_tokenize(tokenizer, text_adv[0]))

        prob, idx = torch.max(predicted_adv, 1) 
        percision=(idx==label).float().mean()
        
        if torch.cuda.is_available():
            accuracy.append(percision.data.item() )
        else:
            accuracy.append(percision.data.numpy()[0] )
    model.train()


    return np.mean(accuracy)



def evaluation_hotflip_adv(opt, device, model, test_iter, tokenizer):
    model.eval()
    accuracy=[]

    for index,batch in enumerate( test_iter):
        text = batch[0].to(device)
        label = batch[1].to(device)
        text_like_syn= batch[6].to(device)
        text_like_syn_valid= batch[7].to(device)

        batch_size = len(text)

        batch_size = len(text)
        if index*batch_size > 200:
            break

        attack_type_dict = {
            'num_steps': opt.test_attack_iters,
            'loss_func': 'ce',
        }
        text_adv = model(mode="get_adv_hotflip", input=text, label=label, text_like_syn_valid=text_like_syn_valid, text_like_syn=text_like_syn, attack_type_dict=attack_type_dict)
        
        predicted_adv = model(mode="text_to_logit", input=text_adv)

        #print("_________________________________")
        #print(inverse_tokenize(tokenizer, text[0]))
        #print(inverse_tokenize(tokenizer, text_adv[0]))

        prob, idx = torch.max(predicted_adv, 1) 
        percision=(idx==label).float().mean()
        
        if torch.cuda.is_available():
            accuracy.append(percision.data.item() )
        else:
            accuracy.append(percision.data.numpy()[0] )
    model.train()


    return np.mean(accuracy)

def inverse_tokenize(tokenizer, tokenized):
    result = tokenizer.sequences_to_texts([tokenized.cpu().numpy()])
    return result[0]

def getOptimizer(params,name="adam",lr=1,weight_decay=1e-4, momentum=None,scheduler=None):
    
    name = name.lower().strip()          
        
    if name=="adadelta":
        optimizer=torch.optim.Adadelta(params, lr=1.0*lr, rho=0.9, eps=1e-06, weight_decay=weight_decay).param_groups()
    elif name == "adagrad":
        optimizer=torch.optim.Adagrad(params, lr=0.01*lr, lr_decay=0, weight_decay=weight_decay)
    elif name == "sparseadam":        
        optimizer=torch.optim.SparseAdam(params, lr=0.001*lr, betas=(0.9, 0.999), eps=1e-08)
    elif name =="adamax":
        optimizer=torch.optim.Adamax(params, lr=0.002*lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif name =="asgd":
        optimizer=torch.optim.ASGD(params, lr=0.01*lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=weight_decay)
    elif name == "lbfgs":
        optimizer=torch.optim.LBFGS(params, lr=1*lr, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
    elif name == "rmsprop":
        optimizer=torch.optim.RMSprop(params, lr=0.01*lr, alpha=0.99, eps=1e-08, weight_decay=weight_decay, momentum=0, centered=False)
    elif name =="rprop":
        optimizer=torch.optim.Rprop(params, lr=0.01*lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
    elif name =="sgd":
        #optimizer=torch.optim.SGD(params, lr=lr, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=False)
        optimizer=torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name =="adam":
        #optimizer=torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        optimizer=torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        print("undefined optimizer, use adam in default")
        optimizer=torch.optim.Adam(params, lr=0.1*lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    
    if scheduler is not None:
        if scheduler == "lambdalr":
            lambda1 = lambda epoch: epoch // 30
            lambda2 = lambda epoch: 0.95 ** epoch
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        elif scheduler=="steplr":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler =="multisteplr":
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        elif scheduler =="reducelronplateau":
            return  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        else:
            pass

    else:
        return optimizer  
    return 

def get_lr_scheduler(name):
    # todo 
    return None
    
    
    
def getLogger():
    import random
    random_str = str(random.randint(1,10000))
    
    now = int(time.time()) 
    timeArray = time.localtime(now)
    timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
    log_filename = "log/" +time.strftime("%Y%m%d", timeArray)
    
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program) 
    if not os.path.exists("log"):
        os.mkdir("log")
    if not os.path.exists(log_filename):
        os.mkdir(log_filename)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',datefmt='%a, %d %b %Y %H:%M:%S',filename=log_filename+'/qa'+timeStamp+"_"+ random_str+'.log',filemode='w')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    return logger

def parse_grid_parameters(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    config_common = config['COMMON']
    dictionary = {}
    for key,value in config_common.items():
        array = value.split(';')
        is_numberic = re.compile(r'^[-+]?[0-9.]+$')
        new_array = []
    
        for value in array:
            value = value.strip()
            result = is_numberic.match(value)
            if result:
                if type(eval(value)) == int:
                    value= int(value)
                else :
                    value= float(value)
            new_array.append(value)
        dictionary[key] = new_array
    return dictionary

def is_writeable(path, check_parent=False):
    '''
    Check if a given path is writeable by the current user.
    :param path: The path to check
    :param check_parent: If the path to check does not exist, check for the
    ability to write to the parent directory instead
    :returns: True or False
    '''
    if os.access(path, os.F_OK) and os.access(path, os.W_OK):
    # The path exists and is writeable
        return True
    if os.access(path, os.F_OK) and not os.access(path, os.W_OK):
    # The path exists and is not writeable
        return False
    # The path does not exists or is not writeable
    if check_parent is False:
    # We're not allowed to check the parent directory of the provided path
        return False
    # Lets get the parent directory of the provided path
    parent_dir = os.path.dirname(path)
    if not os.access(parent_dir, os.F_OK):
    # Parent directory does not exit
        return False
    # Finally, return if we're allowed to write in the parent directory of the
    # provided path
    return os.access(parent_dir, os.W_OK)
def is_readable(path):
    '''
    Check if a given path is readable by the current user.
    :param path: The path to check
    :returns: True or False
    '''
    if os.access(path, os.F_OK) and os.access(path, os.R_OK):
    # The path exists and is readable
        return True
    # The path does not exist
    return False

