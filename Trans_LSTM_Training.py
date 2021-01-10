#!/usr/bin/python
import sys
import os
import subprocess
from os.path import join, isdir

import torch

#Loading the Parser and default arguments
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/TRANS_LM_V1')
import Transformer_lm_arg
from Transformer_lm_arg import parser
args = parser.parse_args()

#************************
import Set_gpus
from Set_gpus import Set_gpu
if args.gpu:
    Set_gpu()
#************************

import numpy as np
from numpy.random import permutation
import fileinput
from torch import autograd, nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from random import shuffle
import matplotlib
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
matplotlib.pyplot.viridis()
os.environ['PYTHONUNBUFFERED'] = '1'
import glob
from statistics import mean
import json
import kaldi_io


#save architecture for decoding
model_path_name=join(args.model_dir,'model_architecture_')
with open(model_path_name, 'w') as f:
    json.dump(args.__dict__, f, indent=2)

#=================================================================
from Dataloader_for_lm import DataLoader
from Load_sp_model import Load_sp_models
from Training_loop_Trans import train_val_model

#from Initializing_RNNLM_model_args import Initialize_RNNLM_model
from Initializing_Trans_model_args import Initialize_Trans_model

from utils__ import weights_init,reduce_learning_rate,read_as_list,gaussian_noise,plotting
from user_defined_losses import preprocess,compute_cer
#===================================================================
if not isdir(args.model_dir):
        os.makedirs(args.model_dir)

png_dir=args.model_dir+'_png'
if not isdir(png_dir):
        os.makedirs(png_dir)
############################################
#================================================================================
#=======================================================
def main():  
        ##
        
        model,optimizer = Initialize_Trans_model(args)
        # Load setpiece models for Dataloaders
        Word_model = Load_sp_models(args.Word_model_path)
        Char_model = Load_sp_models(args.Char_model_path)
        # initilize the model
        #============================================================
        #------------------------------------------------------------ 

        train_gen = DataLoader(files=glob.glob(args.train_path + "*"),
                                max_batch_label_len = args.max_batch_label_len,
                                max_batch_len = args.max_batch_len,
                                max_label_len = args.max_label_len,
                                Word_model = Word_model)   
    

        dev_gen = DataLoader(files=glob.glob(args.dev_path + "*"),
                                max_batch_label_len = args.max_batch_label_len,
                                max_batch_len = args.max_batch_len,
                                max_label_len = 3000,
                                Word_model = Word_model)   

        val_history=np.zeros(args.nepochs)   
        #======================================
        for epoch in range(args.nepochs):

            ##start of the epoch
            tr_CER=[]; tr_BPE_CER=[]; L_train_cost=[];

            # To start the weight Noise from the start
            if args.weight_noise_flag==2:
               weight_noise_flag=True
            
            #
            model.train();
            for trs_no in range(args.validate_interval):
                B1 = train_gen.next()
                assert B1 is not None, "None should never come out of the DataLoader"
                Output_trainval_dict = train_val_model(args = args,
                                                    smp_no = trs_no,
                                                    model = model,
                                                    optimizer = optimizer,
                                                    data_dict = B1,
                                                    trainflag = True)

                #get the losses form the dict
                SMP_CE=Output_trainval_dict.get('cost_cpu')
                L_train_cost.append(SMP_CE)
                #==========================================
                if ((trs_no%args.tr_disp==0) and (trs_no!=0)):
                    print("tr ep:==:>",epoch,"sampl no:==:>",trs_no,"train_cost==:>",mean(L_train_cost),L_train_cost[-1],flush=True)    
                    #------------------------           
            ###validate the model
            #=======================================================
            model.eval()
            #=======================================================
            Vl_CER=[]; Vl_BPE_CER=[];L_val_cost=[]
            val_examples=0
            for vl_smp in range(args.max_val_examples):
                B1 = dev_gen.next()
                smp_feat = B1.get('smp_word_label')
                val_examples+=smp_feat.shape[0]
                assert B1 is not None, "None should never come out of the DataLoader"

                ##brak when the examples are more
                if (val_examples >= args.max_val_examples):
                    break;
                #--------------------------------------                
                Val_Output_trainval_dict=train_val_model(args=args,
                                                        smp_no = vl_smp,
                                                        model = model,
                                                        optimizer = optimizer,
                                                        data_dict = B1,
                                                        trainflag = False)
                

                SMP_CE=Val_Output_trainval_dict.get('cost_cpu')
                L_val_cost.append(SMP_CE)
                #======================================================
                if ((vl_smp%args.vl_disp==0) and vl_smp!=0) or (val_examples==args.max_val_examples-1):
                    print("val epoch:==:>",epoch,"val smp no:==:>",vl_smp,"val_cost:==:>",mean(L_val_cost),L_val_cost[-1],flush=True)                            
            #----------------------------------------------------
            #==================================================================
            val_history[epoch]=(mean(L_val_cost))
            print("val_history:",val_history[:epoch+1])
            
            #saving_weights 
            ct="model_epoch_"+str(epoch)+"_sample_"+str(trs_no)+"_"+str(mean(L_train_cost))+"___"+str(mean(L_val_cost))
            print(ct)
            torch.save(model.state_dict(),join(args.model_dir,str(ct)))
            torch.save(optimizer.state_dict(),join(args.model_dir,str(ct)+'_opt'))                

            ###open the file write and close it to avoid delays
            with open(args.weight_text_file,'a+') as weight_saving_file:
                print(join(args.model_dir,str(ct)), file=weight_saving_file)

            with open(args.Res_text_file,'a+') as Res_saving_file:
                print(float(mean(L_val_cost)), file=Res_saving_file)
            
            if args.early_stopping:
                 A=val_history
                 Non_zero_loss=A[A>0]
                 min_cpts=np.argmin(Non_zero_loss)
                 Non_zero_len=len(Non_zero_loss)

                 if ((Non_zero_len-min_cpts) > 1):                
                    #start regularization only when model starts to overfit
                    weight_noise_flag=True
                    spec_aug_flag=True
                    
                 if (Non_zero_len-min_cpts) > args.early_stopping_patience:
                    print("The model is early stopping........","minimum value of model is:",min_cpts)
                    exit(0)            

#=============================================================================================
if __name__ == '__main__':
    main()



