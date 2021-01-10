#! /usr/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb 
from torch.autograd import Variable
import sys

sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/TRANS_LM_V1')

from Trans_utilities import get_attn_key_pad_mask, get_subsequent_mask, get_attn_pad_mask_encoder, get_attn_pad_mask,get_encoder_non_pad_mask, get_decoder_non_pad_mask,pad_list
from Trans_MHA import MultiHeadAttention, PositionwiseFeedForward, PositionalEncoding
import sys
from Load_sp_model import Load_sp_models
from CrossEntropyLabelSmooth import CrossEntropyLabelSmooth as cal_performance


#============================================================================================================================

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1,ff_dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout) 
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=ff_dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        #
        x=dec_input
        nx=self.norm1(x)
        dec_output, dec_slf_attn = self.slf_attn(nx, nx,nx, mask=slf_attn_mask)
        x=x+self.dropout(dec_output)

        #
        nx=self.norm3(x)
        dec_output = self.pos_ffn(nx)
        dec_output=x+self.dropout(dec_output)
        #
        dec_enc_attn=dec_slf_attn
        return dec_output, dec_slf_attn, dec_enc_attn

#====================================================
#----------------------------------------------------
class Trans_LM(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(self,args):
        super(Trans_LM, self).__init__()
        #------------------------------------------------------------ 
        self.use_gpu = args.gpu
        self.use_pos_emb = int(args.use_pos_emb)
        
        ####word model
        self.Word_model = Load_sp_models(args.Word_model_path)

        self.targets_no = int(self.Word_model.__len__())
        self.pad_index  = self.targets_no       
        self.sos_id     = self.targets_no + 1 # Start of Sentence
        self.eos_id     = self.targets_no + 2 # End of Sentence
        self.mask_id    = self.targets_no + 3
        self.Wout_size  = self.targets_no + 4
        self.word_unk   = self.Word_model.unk_id()
        self.Word_SIL_tok   = self.Word_model.EncodeAsIds('_____')[0]
        self.IGNORE_ID = self.pad_index
        #---------------------------------------------------------------
        

        # parameters
        self.n_tgt_vocab = self.Wout_size
        self.d_word_vec = args.dec_embd_vec_size
        self.n_layers = args.decoder_layers
        self.n_head = args.decoder_heads
        self.d_model = args.decoder_dmodel
        self.d_inner = args.decoder_dinner
        self.dropout = args.decoder_dropout

        self.ff_dropout = args.decoder_ff_dropout
        self.d_k = int(self.d_model/self.n_head)
        self.d_v = int(self.d_model/self.n_head)
        self.tie_dec_emb_weights = args.tie_dec_emb_weights
        self.pe_maxlen = args.pe_max_len

        self.tgt_word_emb = nn.Embedding(self.n_tgt_vocab, self.d_word_vec)

        self.Lin_proj = nn.Linear(self.d_word_vec, self.d_model)
        self.Lin_proj_norm = nn.LayerNorm(self.d_model)


        self.x_scale = math.sqrt(self.d_model)       

        self.positional_encoding = PositionalEncoding(self.d_model, max_len=self.pe_maxlen, dropout=self.dropout)
        self.output_norm = nn.LayerNorm(self.d_model)
        
        self.layer_stack = nn.ModuleList([DecoderLayer(self.d_model, self.d_inner, self.n_head,
                                                       self.d_k, self.d_v, dropout=self.dropout,
                                                       ff_dropout=self.ff_dropout) for _ in range(self.n_layers)])

        self.tgt_word_prj = nn.Linear(self.d_model, self.n_tgt_vocab)

        ###weight tie-ing 
        ##for the tieing to be possible weights of self.tgt_word_emb, self.tgt_word_prj should be equal 
        ### self.d_word_vec should be equal to self.d_model or a linear layer should be used
        if self.tie_dec_emb_weights and (self.d_model == self.d_word_vec):
                self.tgt_word_emb.weight=self.tgt_word_prj.weight

        ###label_smoothed_cross_entropy
        self.label_smoothing=args.label_smoothing
     

    def preprocess(self, padded_input):
        """Generate decoder input and output label from padded_input
        Add <sos> to decoder input, and add <eos> to decoder output label """

        ys = [y[y != self.IGNORE_ID] for y in padded_input]  # parse padded ys
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos_id])
        sos = ys[0].new([self.sos_id])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys ]
       

        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos_id)
        ys_out_pad = pad_list(ys_out, self.IGNORE_ID) ####original
        
        assert ys_in_pad.size() == ys_out_pad.size()
        return ys_in_pad, ys_out_pad

    def forward(self, padded_input, return_attns=True):
        """ Args: padded_input: N x To encoder_padded_outputs: N x Ti x H   Returns:"""
         
        
        dec_slf_attn_list, dec_enc_attn_list = [], []
        #
        padded_input = padded_input.cuda() if self.use_gpu else padded_input        

        
        #Get Deocder Input and Output
        ys_in_pad, ys_out_pad = self.preprocess(padded_input)
        
        #Prepare masks
        slf_attn_mask_subseq = get_subsequent_mask(ys_in_pad)
        output_length = ys_in_pad.size(1)

        ###Not using other masks, using Justs casual masks
        non_pad_mask = None;
        dec_enc_attn_mask = None;
        slf_attn_mask_keypad = None;
        slf_attn_mask = slf_attn_mask_subseq
        embd_output = self.tgt_word_emb(ys_in_pad)

      
        ###no pos_embeding as said in ACHEN paper
        if self.use_pos_emb:
                dec_output=self.positional_encoding(embd_output)
        else:
                dec_output=embd_output
        
        ###for language model
        #encoder_padded_outputs=dec_output
        

        dec_output = self.Lin_proj_norm(self.Lin_proj(dec_output))
        
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output,
                                                                non_pad_mask=non_pad_mask,
                                                                slf_attn_mask=slf_attn_mask,
                                                                dec_enc_attn_mask=dec_enc_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        # before softmax
        #------------
        dec_output=self.output_norm(dec_output)
        seq_logit = self.tgt_word_prj(dec_output)
        #---------------

        pred, gold = seq_logit, ys_out_pad
        #
        
        cost = cal_performance(pred.view(-1,self.Wout_size), gold.view(-1),self.IGNORE_ID,normalize_length=False,smoothing=self.label_smoothing,Ignore_padding=True)
        numtokens = (gold.ne(self.IGNORE_ID)*1).sum()
        cost=cost/numtokens

        output_dict = {'cost':cost, 'dec_slf_attn_list':dec_slf_attn_list, 'dec_enc_attn_list':dec_enc_attn_list, 'Char_cer':None, 'Word_cer':None}
        return output_dict

#==========================================================================

    def prediction_from_trained_model(self,ys,scores_list,):
                """####this function is accessed from the decoder to get the output from the decoder,
                   and this could be used for model ensembling an
                   ####when this function is called with prediceted label sequences,
                   it gives the proability distribution for the next possible labels roughly this gives P(y_i |y_(i<i)
                """
                non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1) # 1xix1
                slf_attn_mask = get_subsequent_mask(ys)

                # -- Forward
                embd_output = self.tgt_word_emb(ys_in_pad)

                ###no pos_embeding as said in ACHEN paper
                if self.use_pos_emb:
                        dec_output=self.positional_encoding(embd_output)
                else:
                        dec_output=embd_output

                for dec_layer in self.layer_stack:
                    dec_output, _, _ = dec_layer(dec_output,
                                                        encoder_outputs,
                                                        non_pad_mask=None,
                                                        slf_attn_mask=slf_attn_mask,
                                                        dec_enc_attn_mask=None)

                dec_output_Bneck=self.output_norm(dec_output)
                dec_output=self.output_norm(dec_output[:, -1])
                seq_logit = self.tgt_word_prj(dec_output)

                scores_list.append(seq_logit.unsqueeze(1))
                local_scores = F.log_softmax(seq_logit, dim=1)
                scores = F.softmax(seq_logit, dim=1)
                present_label=torch.argmax(scores, dim=1)
                return local_scores,scores_list,present_label,dec_output_Bneck

#=============================================================================================================
