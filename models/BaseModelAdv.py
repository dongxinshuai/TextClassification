# -*- coding: utf-8 -*-
import os
import numpy as np

from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .CNNKim import *

class KIMCNN1D_Adv(KIMCNN1D):
    def __init__(self, opt ):
        super(KIMCNN1D_Adv, self).__init__(opt)

        # inverse embedding
        print("making inverse embedding")
        """
        inverse_embedding_weight = self.embedding.weight.detach().cpu()
        inverse_embedding_weight = inverse_embedding_weight.numpy()
        inverse_embedding_weight = np.matrix(inverse_embedding_weight)
        inverse_embedding_weight = np.linalg.pinv(inverse_embedding_weight)
        inverse_embedding_weight = torch.FloatTensor(inverse_embedding_weight)
        """
        #self.inverse_embedding = nn.Embedding(self.embedding_dim, self.vocab_size + 2, ) 
        self.inverse_embedding = nn.Linear( self.vocab_size + 2,self.embedding_dim, bias=False)
        """
        self.inverse_embedding.weight=nn.Parameter(inverse_embedding_weight)            
        self.inverse_embedding.weight.requires_grad = False
        """
        
        # embd to embdnew
        self.new_embedding_dim = self.embedding_dim
        self.linear_transform_embd = nn.Linear(self.embedding_dim, self.new_embedding_dim)

        # embdnew to embd
        self.inverse_linear_transform_embd = nn.Linear(self.new_embedding_dim, self.embedding_dim)
        #self.update_linear_transform_embd()

        # l2 radius 
        self.word_synonym_radius = nn.Embedding(self.vocab_size + 2, 1) #, padding_idx=self.vocab_size + 1
        self.word_synonym_radius.weight.requires_grad = True

    def update_linear_transform_embd(self):
        weight=self.inverse_linear_transform_embd.weight.detach().cpu() # W x +b 
        bias=self.inverse_linear_transform_embd.bias.detach().cpu()

        weight = weight.numpy()
        weight = np.matrix(weight)
        weight = np.linalg.pinv(weight)
        weight = torch.FloatTensor(weight)

        self.linear_transform_embd.weight=nn.Parameter(weight)  
        bias = -bias.reshape(1,-1).mm(weight.transpose(0,1)) 
        bias = bias.squeeze()
        self.linear_transform_embd.bias=nn.Parameter(bias) 

    def update_inverse_linear_transform_embd(self):
        weight=self.linear_transform_embd.weight.detach().cpu() # W x +b 
        bias=self.linear_transform_embd.bias.detach().cpu()

        weight = weight.numpy()
        weight = np.matrix(weight)
        weight = np.linalg.pinv(weight)
        weight = torch.FloatTensor(weight)

        self.inverse_linear_transform_embd.weight=nn.Parameter(weight)  
        bias = -bias.reshape(1,-1).mm(weight.transpose(0,1)) 
        bias = bias.squeeze()
        self.inverse_linear_transform_embd.bias=nn.Parameter(bias) 


    def loss_radius(self, roots, synonyms):
        embd_roots = self.text_to_embd(roots.reshape(-1,1))
        embd_synonyms = self.text_to_embd(synonyms)
        n, len_syn_set, embd_dim = embd_synonyms.shape
        delta = (embd_synonyms-embd_roots).reshape(-1, embd_dim)

        radius = torch.nn.functional.pairwise_distance(delta, torch.zeros_like(delta), p=2.0) # n*len_syn_set,
        max_radius, _ = radius.reshape(n, len_syn_set).max(-1)
        
        for i in range(n):
            self.word_synonym_radius.weight[roots[i],0]=max_radius[i]
        
        print(max_radius.mean())

        #saved_radius = self.word_synonym_radius(roots).squeeze() # n
        #return torch.nn.functional.mse_loss(max_radius, saved_radius, reduction='mean')

    def l2_project(self, grad):

        torch_small_constant = 1e-12*torch.ones_like(grad)

        grad_norm=grad*grad
        grad_norm=torch.sum(grad_norm, dim=-1, keepdim=True)
        grad_norm = torch.sqrt(grad_norm)
        grad_norm = torch.max(torch_small_constant, grad_norm)
        grad = grad / grad_norm

        return grad

    def l2_clip(self, pert, eps):
    
        torch_small_constant = 1e-12*torch.ones_like(pert)
        torch_ones = torch.ones_like(pert)

        pert_norm=pert*pert
        pert_norm=torch.sum(pert_norm, dim=-1, keepdim=True)
        pert_norm = torch.sqrt(pert_norm)
        pert_norm = torch.max(torch_small_constant, pert_norm)
        ratio = eps / pert_norm
        ratio = torch.min(torch_ones, ratio)

        return pert*ratio

    def get_embd_adv(self, embd, y, attack_type_dict):
        
        # record context
        self_training_context = self.training
        # set context
        self.eval()

        device = embd.device
        # get param of attacks

        num_steps=attack_type_dict['num_steps']
        step_size=attack_type_dict['step_size']
        random_start=attack_type_dict['random_start']
        epsilon=attack_type_dict['epsilon']
        loss_func=attack_type_dict['loss_func']
        direction=attack_type_dict['direction']

        batch_size=len(embd)

        embd_ori = embd
        
        # random start
        if random_start:
            embd_adv = embd_ori.detach() + 0.001 * torch.randn(embd_ori.shape).to(device).detach()
        else:
            embd_adv = embd_ori.detach()


        for _ in range(num_steps):
            embd_adv.requires_grad_()
            grad = 0
            with torch.enable_grad():
                if loss_func=='ce':
                    logit_adv = self.embd_to_logit(embd_adv)
                    if direction == "towards":
                        loss = -F.cross_entropy(logit_adv, y, reduction='sum')
                    elif direction == "away":
                        loss = F.cross_entropy(logit_adv, y, reduction='sum')
                grad = torch.autograd.grad(loss, [embd_adv])[0]

            grad=self.l2_project(grad)

            embd_adv = embd_adv.detach() + step_size * grad.detach()
            
            perturbation = self.l2_clip(embd_adv-embd_ori, epsilon)
            embd_adv = embd_ori.detach() + perturbation.detach()
            
        # resume context
        if self_training_context == True:
            self.train()
        else:
            self.eval()

        return embd_adv.detach()

    def embd_to_embdnew(self,embd):
        embdnew = self.linear_transform_embd(embd)
        return embdnew

    def embd_to_text(self, embd):
        bs, sen_len, embd_dim = embd.shape
        text = embd.reshape(-1, embd_dim).mm(self.inverse_embedding.weight)
        return text.reshape(bs, sen_len, -1)
        
    def embdnew_to_embd(self, embdnew):
        embd = self.inverse_linear_transform_embd(embdnew)
        return embd

    def loss_text_adv(self, input, label):
        #p = -F.log_softmax(input, dim=-1)
        #loss = p*(label.to(p.dtype)).sum()
        input_shape = input.shape
        return F.cross_entropy(input.reshape(-1,input_shape[-1]), label.reshape(-1) )

    def text_to_radius(self, inp):
        saved_radius = self.word_synonym_radius(inp) # n, len, 1
        return saved_radius.detach()

    def forward(self, mode, input, label=None, attack_type_dict=None):
        if mode == "get_embd_adv":
            assert(attack_type_dict is not None)
            out = self.get_embd_adv(input, label, attack_type_dict)
        if mode == "embd_to_logit":
            out = self.embd_to_logit(input)
        if mode == "text_to_embd":
            out = self.text_to_embd(input)
        if mode == "text_to_radius":
            out = self.text_to_radius(input)
        if mode == "text_to_logit":
            embd = self.text_to_embd(input)
            #embd = self.embd_to_embdnew(embd)
            out = self.embd_to_logit(embd)
        if mode == "embd_to_embdnew":
            out = self.embd_to_embdnew(input)
        if mode == "embd_to_text":
            out = self.embd_to_text(input)
        if mode == "embdnew_to_embd":
            out = self.embdnew_to_embd(input)
        if mode == "update_inverse_linear_transform_embd":
            self.update_inverse_linear_transform_embd()
            out = None
        if mode == "update_linear_transform_embd":
            self.update_linear_transform_embd()
            out = None
        if mode == "loss_text_adv":
            out = self.loss_text_adv(input, label)
        if mode == "loss_radius":
            out = self.loss_radius(input, label)
        return out


