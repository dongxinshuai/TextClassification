# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertModel,BertForSequenceClassification
from models.BaseModelAdv import AdvBaseModel
import os

class BertModel_forward_modified(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if input_ids is not None:
            embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
            )
            return embedding_output

        elif inputs_embeds is not None:
            embedding_output = inputs_embeds
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

            assert(not return_dict)
            return (sequence_output, pooled_output) + encoder_outputs[1:]



class AdvBERT(AdvBaseModel): 
    def __init__(self, opt ):
        super(AdvBERT, self).__init__(opt, is_bert=True)
        #self.opt=opt

        self.bert_model = torch.nn.DataParallel(BertModel_forward_modified.from_pretrained('bert-base-uncased'))
        #self.bert_model = BertModel_forward_modified.from_pretrained('bert-base-uncased')
        #self.bert_model_for_embd = BertModel_forward_modified.from_pretrained('bert-base-uncased')
        #self.bert_model = BertModel_add_text2embd.from_pretrained('bert-base-uncased')

        #for param in self.bert_model_for_embd.parameters():
        #    param.requires_grad=False
        #for param in self.bert_model_for_embd.embeddings.parameters():
        #    param.requires_grad=True
        for param in self.bert_model.parameters():
            param.requires_grad=True
        self.hidden2label = nn.Linear(768, opt.label_size)

        self.eval_adv_mode = True

    #def forward(self,  content):
    #    _, pooled = self.bert_model(content)
    #    logits = self.hidden2label(pooled)
    #    return logits
    

    def embd_to_logit(self, embd, attention_mask):
        _, pooled = self.bert_model(inputs_embeds=embd, attention_mask=attention_mask)
        logits = self.hidden2label(pooled)
        return logits
        
    def text_to_embd(self, input_ids, token_type_ids):
        embedding_output = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids)
        return embedding_output

    #def text_to_embd(self, input_ids, token_type_ids):
        #embedding_output = self.bert_model_for_embd.embeddings(
        #    input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=None)
        #return embedding_output
    
    def get_adv_by_convex_syn(self, embd, y, syn, syn_valid, text_like_syn, attack_type_dict, bert_mask, text_for_vis, record_for_vis):
        
        # record context
        self_training_context = self.training
        # set context
        if self.eval_adv_mode:
            self.eval()
        else:
            self.train()

        device = embd.device
        # get param of attacks

        num_steps=attack_type_dict['num_steps']
        loss_func=attack_type_dict['loss_func']
        w_optm_lr=attack_type_dict['w_optm_lr']
        sparse_weight = attack_type_dict['sparse_weight']
        out_type = attack_type_dict['out_type']

        batch_size, text_len, embd_dim = embd.shape
        batch_size, text_len, syn_num, embd_dim = syn.shape

        w = torch.empty(batch_size, text_len, syn_num, 1).to(device).to(embd.dtype)
        #ww = torch.zeros(batch_size, text_len, syn_num, 1).to(device).to(embd.dtype)
        #ww = ww+500*(syn_valid.reshape(batch_size, text_len, syn_num, 1)-1)
        nn.init.kaiming_normal_(w)
        w.requires_grad_()
        
        import utils
        params = [w] 
        optimizer = utils.getOptimizer(params,name='adam', lr=w_optm_lr,weight_decay=2e-5)

        def get_comb_p(w, syn_valid):
            ww=w*syn_valid.reshape(batch_size, text_len, syn_num, 1) + 500*(syn_valid.reshape(batch_size, text_len, syn_num, 1)-1)
            return F.softmax(ww, -2)

        def get_comb_ww(w, syn_valid):
            ww=w*syn_valid.reshape(batch_size, text_len, syn_num, 1) + 500*(syn_valid.reshape(batch_size, text_len, syn_num, 1)-1)
            return ww

        def get_comb(p, syn):
            return (p* syn.detach()).sum(-2)


        embd_ori=embd.detach()
        with torch.no_grad():
            logit_ori = self.embd_to_logit(embd_ori, bert_mask)

        for _ in range(num_steps):
            optimizer.zero_grad()
            with torch.enable_grad():
                ww = get_comb_ww(w, syn_valid)
                #comb_p = get_comb_p(w, syn_valid)
                embd_adv = get_comb(F.softmax(ww, -2), syn)
                if loss_func=='ce':
                    logit_adv = self.embd_to_logit(embd_adv, bert_mask)
                    loss = -F.cross_entropy(logit_adv, y, reduction='sum')
                elif loss_func=='kl':
                    logit_adv = self.embd_to_logit(embd_adv, bert_mask)
                    criterion_kl = nn.KLDivLoss(reduction="sum")
                    loss = -criterion_kl(F.log_softmax(logit_adv, dim=1),
                                        F.softmax(logit_ori.detach(), dim=1))

                #print("ad loss:", loss.data.item())
                                    
                if sparse_weight !=0:
                    #loss_sparse = (comb_p*comb_p).mean()
                    loss_sparse = (-F.softmax(ww, -2)*F.log_softmax(ww, -2)).sum(-2).mean()
                    #loss -= sparse_weight*loss_sparse
                    
                    loss = loss + sparse_weight*loss_sparse
                    #print(loss_sparse.data.item())

            #loss*=1000
            loss.backward()
            optimizer.step()

        #print((ww-w).max())

        comb_p = get_comb_p(w, syn_valid)

        if self.opt.vis_w_key_token is not None:
            assert(text_for_vis is not None and record_for_vis is not None)
            vis_n, vis_l = text_for_vis.shape
            for i in range(vis_n):
                for j in range(vis_l):
                    if text_for_vis[i,j] == self.opt.vis_w_key_token:
                        record_for_vis["comb_p_list"].append(comb_p[i,j].cpu().detach().numpy())
                        record_for_vis["embd_syn_list"].append(syn[i,j].cpu().detach().numpy())
                        record_for_vis["syn_valid_list"].append(syn_valid[i,j].cpu().detach().numpy())
                        record_for_vis["text_syn_list"].append(text_like_syn[i,j].cpu().detach().numpy())
                        
                        print("record for vis", len(record_for_vis["comb_p_list"]))
                    if len(record_for_vis["comb_p_list"])>=300:
                        dir_name = self.opt.resume.split(self.opt.model)[0]
                        file_name = self.opt.dataset+"_vis_w_"+str(self.opt.attack_sparse_weight)+"_"+str(self.opt.vis_w_key_token)+".pkl"
                        file_name = os.path.join(dir_name, file_name)
                        f=open(file_name,'wb')
                        pickle.dump(record_for_vis, f)
                        f.close()
                        sys.exit()
                        

        if out_type == "text":
            # need to be fix, has potential bugs. the trigger dependes on data.
            assert(text_like_syn is not None) # n l synlen
            comb_p = comb_p.reshape(batch_size* text_len, syn_num)
            ind = comb_p.max(-1)[1] # shape batch_size* text_len
            out = (text_like_syn.reshape(batch_size* text_len, syn_num)[np.arange(batch_size*text_len), ind]).reshape(batch_size, text_len)
        elif out_type == "comb_p":
            out = comb_p

        # resume context
        if self_training_context == True:
            self.train()
        else:
            self.eval()

        return out.detach()

    def forward(self, mode, input, comb_p = None, label=None, text_like_syn_embd=None, text_like_syn_valid=None, text_like_syn=None, attack_type_dict=None, bert_mask=None, bert_token_id=None, text_for_vis=None, record_for_vis=None):
        if mode == "get_embd_adv":
            assert(attack_type_dict is not None)
            out = self.get_embd_adv(input, label, attack_type_dict)
        if mode == "get_adv_by_convex_syn":
            assert(attack_type_dict is not None)
            assert(text_like_syn_embd is not None)
            assert(text_like_syn_valid is not None)
            out = self.get_adv_by_convex_syn(input, label, text_like_syn_embd, text_like_syn_valid, text_like_syn, attack_type_dict, bert_mask, text_for_vis, record_for_vis)
        if mode == "embd_to_logit":
            out = self.embd_to_logit(input, bert_mask)
        if mode == "text_to_embd":
            out = self.text_to_embd(input, bert_token_id)
        if mode == "text_to_logit":
            embd = self.text_to_embd(input, bert_token_id)
            out = self.embd_to_logit(embd, bert_mask)
        if mode == "text_syn_p_to_logit":
            assert(comb_p is not None)
            bs, tl, sl = input.shape
            text_like_syn_embd = self.text_to_embd(input.reshape(bs*tl, sl), bert_token_id.reshape(bs,tl,1).repeat(1,1,sl).reshape(bs*tl, sl)).reshape(bs, tl, sl, -1)
            embd = (comb_p*text_like_syn_embd).sum(-2)
            out = self.embd_to_logit(embd, bert_mask)

        return out




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

