# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np



from .LSTM import LSTMClassifier
from .CNNBasic import BasicCNN1D,BasicCNN2D, AdvBasicCNN1D, AdvBasicCNN2D
from .CNNKim import KIMCNN1D,KIMCNN2D
from .CNNMultiLayer import MultiLayerCNN
from .CNNInception import InceptionCNN
from .FastText import FastText
from .Capsule import CapsuleNet
from .RCNN import RCNN
from .RNN_CNN import RNN_CNN
from .LSTMBI import LSTMBI, AdvLSTMBI
from .Transformer import AttentionIsAllYouNeed
from .SelfAttention import SelfAttention
from .LSTMwithAttention import LSTMAttention
#from .BERTFast import BERTFast
from .BERT import AdvBERT
from .ForSnli import AdvDecAtt, AdvDecAtt_FromCert, AdvBOW, AdvEntailmentCNN
#from .BaseModelAdv import  KIMCNN1D_Adv
def setup(opt):
    
    if opt.model == 'lstm':
        model = LSTMClassifier(opt)
    elif opt.model == 'basic_cnn' or opt.model == "cnn":
        model = BasicCNN1D(opt)
    elif opt.model == 'ent_cnn_adv':
        model = AdvEntailmentCNN(opt)
    elif opt.model == 'cnn_adv':
        #model = KIMCNN1D_Adv(opt)
        model = AdvBasicCNN1D(opt)
    elif opt.model == 'cnn_2d_adv':
        model = AdvBasicCNN2D(opt)
    elif opt.model == 'decomp_att_adv':
        model = AdvDecAtt_FromCert(opt)
    elif opt.model == 'bow_adv':
        model = AdvBOW(opt)
    elif opt.model ==  'bilstm_adv':
        model = AdvLSTMBI(opt)
    elif opt.model == 'baisc_cnn_2d' :
        model = BasicCNN2D(opt)
    elif opt.model == 'kim_cnn' :
        model = KIMCNN1D(opt)
    elif opt.model ==  'kim_cnn_2d':
        model = KIMCNN2D(opt)
    elif opt.model ==  'multi_cnn':
        model = MultiLayerCNN(opt)
    elif opt.model ==  'inception_cnn':
        model = InceptionCNN(opt) 
    elif opt.model ==  'fasttext':
        model = FastText(opt)
    elif opt.model ==  'capsule':
        model = CapsuleNet(opt)
    elif opt.model ==  'rnn_cnn':
        model = RNN_CNN(opt)
    elif opt.model ==  'rcnn':
        model = RCNN(opt)
    elif opt.model ==  'bilstm':
        model = LSTMBI(opt)
    elif opt.model == "transformer":
        model = AttentionIsAllYouNeed(opt)
    elif opt.model == "selfattention":
        model = SelfAttention(opt)
    elif opt.model == "lstm_attention":
        model =LSTMAttention(opt)
    #elif opt.model == "bertfast":
    #    model =BERTFast(opt)
    elif opt.model == "bert_adv":
        model =AdvBERT(opt)
    else:
        raise Exception("model not supported: {}".format(opt.model))
    return model
