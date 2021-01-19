#!/usr/bin/python
import sys
import os
import subprocess

from os.path import join, isdir, isfile
import torch
import json

import numpy as np
from torch import autograd, nn, optim
import torch.nn.functional as F

from argparse import Namespace

#Loading the Parser and default arguments
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/TRANS_LM_V1')
import Transformer_lm_arg
from Transformer_lm_arg import parser
args = parser.parse_args()


#=================================================================
from Load_sp_model import Load_sp_models
from Initializing_Trans_model_args import Initialize_Trans_model
from get_best_weights import get_best_weights
from Stocasting_Weight_Addition import Stocasting_Weight_Addition

#===================================================================
args = parser.parse_args()
model_path_name=join(args.model_dir,'model_architecture_')
#--------------------------------
###load the architecture if you have to load
with open(model_path_name, 'r') as f:
        TEMP_args = json.load(f)

ns = Namespace(**TEMP_args)
args=parser.parse_args(namespace=ns)

###make SWA name 
model_name = str(args.model_dir).split('/')[-1]
ct=model_name+'_SWA_random_tag_'+str(args.SWA_random_tag)
args.gpu=0
args.pre_trained_weight="0"
model,optimizer=Initialize_Trans_model(args)
##check the Weight averaged file and if the file does not exist then lcreate them
## if the file exists load them
model_names,checkpoint_ter = get_best_weights(args.weight_text_file, args.Res_text_file)
model_names_checkpoints=model_names[:args.early_stopping_checkpoints]
swa_files=model_name+'_SWA_random_tag_weight_files_'+str(args.SWA_random_tag) + '_args_ealystpping_checkpoints_'+str(args.early_stopping_checkpoints)
outfile=join(args.model_dir,swa_files)

#-----------
with open(outfile,'a+') as outfile:
	print(model_names_checkpoints,file=outfile)

#-----------
#print(model,model_names_checkpoints)
model = Stocasting_Weight_Addition(model, model_names_checkpoints)
torch.save(model.state_dict(),join(args.model_dir,ct))
#return model

#print(join(args.model_dir,ct))

## ##load the required weights 
#args.pre_trained_weight = join(args.model_dir,str(ct))
#model,optimizer=Initialize_Trans_model(args)
#model.eval()
