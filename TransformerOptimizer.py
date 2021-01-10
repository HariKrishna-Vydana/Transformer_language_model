#!/usr/bin/python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable








#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class TransformerOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, k, d_model, step_num, warmup_steps=4000):
        self.optimizer = optimizer
        self.k = k

        #present_lr=[param_group['lr'] for param_group in self.optimizer.param_groups]
        self.init_lr = d_model ** (-0.5)
        self.warmup_steps = warmup_steps
        self.step_num = step_num
        self.reduction_factor=1
    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        lr = self.k * self.init_lr * min(self.step_num ** (-0.5),
                                         self.step_num * (self.warmup_steps ** (-1.5)))

        #print(lr,self.step_num ** (-0.5),self.step_num * self.warmup_steps ** (-1.5),self.reduction_factor)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def set_k(self, k):
        self.k = k

    def set_step_num(self, step_num):
        self.step_num=step_num

    def reduce_learning_rate(self, k):
        self.reduction_factor = self.reduction_factor*k
        #print(self.reduction_factor)

    def print_lr(self):
        present_lr=[param_group['lr'] for param_group in self.optimizer.param_groups]
        return present_lr[0]


#==============================================================================================================
#---------------------------------------------------------------------------------------------------------------


