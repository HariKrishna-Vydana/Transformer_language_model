#!/usr/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
from torch.autograd import Variable
import sys

#=============================================================================================================
def  CrossEntropyLabelSmooth(pred, gold,IGNORE_ID,normalize_length,smoothing,Ignore_padding=False):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        
        inputs  = pred
        targets = gold
        num_classes = pred.size(1)

        log_probs = F.log_softmax(inputs,dim=1)
        batch_size=inputs.size()[0]

        ##How scatter works here 
        #torch.zeros(log_probs.size())
        #vecotr.scatter_(dim,ind,value)
        #torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)

        targets = Variable(torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1))
        targets=targets.cuda() if log_probs.is_cuda else targets
        targets = (1 - smoothing) * targets + smoothing/num_classes
        loss = (- targets * log_probs).sum(dim=1)

        #------------------------------------------
        if Ignore_padding:
            ##zero the elements which belong to the padding id in the loss
            ####was needed for speller
            non_pad_mask = gold.ne(IGNORE_ID)
            loss=loss.masked_select(non_pad_mask)
        #------------------------------------------
        loss=loss.sum()

        return loss

