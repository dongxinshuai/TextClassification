# -*- coding: utf-8 -*-
import torch as t
import numpy as np
from torch import nn
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertModel
from models.BaseModel import BaseModel
import os
class BERT(nn.Module): 
    def __init__(self, opt ):
        super(BERT, self).__init__()
        self.opt=opt

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert_model.parameters():
            param.requires_grad=True
        self.hidden2label = nn.Linear(768, opt.label_size)

        self.properties = {"model_name":self.__class__.__name__,
        #                "embedding_dim":self.opt.embedding_dim,
        #                "embedding_training":self.opt.embedding_training,
        #                "max_seq_len":self.opt.max_seq_len,
                        "batch_size":self.opt.batch_size,
                        "learning_rate":self.opt.learning_rate,
                        }

    def forward(self,  content):
        _, pooled = self.bert_model(content)
        logits = self.hidden2label(pooled)
        return logits

    
    def save(self,save_dir="saved_model",metric=None):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.model_info = "__".join([k+"_"+str(v) if type(v)!=list else k+"_"+str(v)[1:-1].replace(",","_").replace(",","")  for k,v in self.properties.items() ])
        if metric:
            path = os.path.join(save_dir, str(metric)[2:] +"_"+ self.model_info)
        else:
            path = os.path.join(save_dir,self.model_info)
        t.save(self,path)
        return path

import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--hidden_dim', type=int, default=128,
                    help='hidden_dim')   
    
    
    parser.add_argument('--batch_size', type=int, default=64,
                    help='batch_size')
    parser.add_argument('--embedding_dim', type=int, default=300,
                    help='embedding_dim')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning_rate')
    parser.add_argument('--grad_clip', type=float, default=1e-1,
                    help='grad_clip')
    parser.add_argument('--model', type=str, default="lstm",
                    help='model name')
    parser.add_argument('--label_size', type=str, default=2,
                    help='label_size')


#
    args = parser.parse_args()
    args.embedding_dim=300
    args.vocab_size=10000
    args.kernel_size=3
    args.num_classes=3
    args.content_dim=256
    args.max_seq_len=50
    
#
#    # Check if args are valid
#    assert args.rnn_size > 0, "rnn_size should be greater than 0"


    return args
 
if __name__ == '__main__':
    

    opt = parse_opt()
    m = BERT(opt)
    content = t.autograd.Variable(t.arange(0,3200).view(-1,50)).long()
    o = m(content)
    print(o.size())

