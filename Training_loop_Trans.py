#!/usr/bin/python

import sys
import os
import torch
#----------------------------------------
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/TRANS_LM_V1')
from utils__ import weights_init,gaussian_noise
#---------------------------------------
def train_val_model(**kwargs):
        args = kwargs.get('args')
        smp_no = kwargs.get('smp_no')

        model = kwargs.get('model')
        optimizer = kwargs.get('optimizer')

        trainflag = kwargs.get('trainflag')
        B1 = kwargs.get('data_dict')
        smp_word_label = B1.get('smp_word_label')
        smp_trans_text = B1.get('smp_trans_text')  

        #################finished expanding the keyword arguments#########
        ###################################################################
        optimizer.zero_grad() 
        Word_target = torch.LongTensor(smp_word_label)

        #--------------------------------
        Decoder_out_dict = model(Word_target)
        cost = Decoder_out_dict.get('cost')
        
        if trainflag and not (torch.isinf(cost).any() or torch.isnan(cost).any()):
                cost.backward()

                if args.clip_grad_norm != 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad_norm)

                cost = cost/args.accm_grad
                cost.detach()

                ###gradient accumilation
                if(smp_no%args.accm_grad)==0:
                    optimizer.step()
                    optimizer.zero_grad()

        elif (torch.isinf(cost).any() or torch.isnan(cost).any()):
                print(B1,cost,smp_word_label,smp_trans_text)
                exit(0)
        else:
                pass;
        #--------------------------------------
        numtokens = Decoder_out_dict.get('numtokens')

        cost_cpu = torch.exp(cost*args.accm_grad) if trainflag else torch.exp(cost)
        cost_cpu = cost_cpu.item()
        #==================================================
        Output_trainval_dict={'cost_cpu':cost_cpu}
        return Output_trainval_dict
#=========================================================
