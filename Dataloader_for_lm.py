#!/usr/bin/python
import kaldi_io
import sys
import os
from os.path import join, isdir
from numpy.random import permutation
import itertools
import keras
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import queue
from threading  import Thread
import random
import glob


sys.path.insert(0, '/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE')
from Load_sp_model import Load_sp_models

#===============================================
#-----------------------------------------------  
class DataLoader(object):

    def __init__(self,files,max_batch_label_len, max_batch_len,max_label_len,Word_model,queue_size=100):
        self.files = files 
        self.Word_model = Word_model
        self.max_batch_label_len = max_batch_label_len
        self.max_batch_len = max_batch_len
        self.Word_padding_id = self.Word_model.__len__()     
        self.word_space_token   = self.Word_model.EncodeAsIds('_____')[0]
        self.max_label_len = max_label_len

        self.queue = queue.Queue(queue_size)
        
        self._reading_finished = False
        self.label_list = {}
        self._thread = Thread(target=self.__load_data)
        self._thread.daemon = True
        self._thread.start()
       
    def __reset_the_data_holders(self):
            self.batch_names=[]

            self.batch_word_labels=[]
            self.batch_word_label_length=[]    
            
            self.batch_word_text=[]
            self.batch_word_text_length=[]

    def __load_data(self):
        self.__reset_the_data_holders()
        max_batch_label_len = self.max_batch_label_len
        while True:
            #print('Finished whole data: Next iteraton of the data---------->')
            random.shuffle(self.files)
            for name in self.files:
                lines=open(name,'r').readlines()
                #print(name)

                for line in lines:
                    line = line.strip()
                    key = line.split(' ')[0] 
                    labels = line.split(' ')[1:]               

                    ####tokernizing the text
                    word_tokens=self.Word_model.EncodeAsIds(" ".join(labels))            
                    #-----------------####case when the data should be thrown away

                    if (len(labels)<1) or (len(labels) > self.max_label_len):
                          continue;

                    total_labels_in_batch=max(max(self.batch_word_label_length,default=0),len(word_tokens))*(len(self.batch_names)+4)
                    if total_labels_in_batch > self.max_batch_label_len or len(self.batch_word_labels)==self.max_batch_len:
                            
                            ##==============================================================
                            #####to clumsy ------->
                            #CCCC=list(zip(batch_data,self.batch_names,batch_labels,self.batch_word_labels,self.batch_word_text,batch_label_length,batch_length,self.batch_word_label_length,self.batch_word_text_length))
                            #random.shuffle(CCCC)
                            #batch_data,self.batch_names,batch_labels,self.batch_word_labels,self.batch_word_text,batch_label_length,batch_length,self.batch_word_label_length,self.batch_word_text_length=zip(*CCCC)
                            ##==============================================================

                            padded_word_labels=pad_sequences(self.batch_word_labels,maxlen=max(self.batch_word_label_length),dtype='int32',padding='post',value=self.Word_padding_id)
                            padded_trans_text=pad_sequences(self.batch_word_text,maxlen=max(self.batch_word_text_length),dtype=object,padding='post',value='')                            
                            batch_data_dict={'batch_names':self.batch_names, 'smp_word_label':padded_word_labels, 'smp_trans_text':padded_trans_text}

                            self.queue.put(batch_data_dict)
                            #==============================
                            self.__reset_the_data_holders()
                        #==============================================================
                    self.batch_names.append(key)
                    self.batch_word_labels.append(word_tokens)
                    self.batch_word_label_length.append(len(word_tokens))

                    self.batch_word_text.append(labels)
                    self.batch_word_text_length.append(len(labels))

    def next(self, timeout=30000):
        return self.queue.get(block=True, timeout=timeout)
#===================================================================




# sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/KAT_Attention')
# import RNNLM_config
# from RNNLM_config import parser
# args = parser.parse_args()
# print(args)

# Word_model=Load_sp_models(args.Word_model_path)
# train_path='/mnt/matylda3/vydana/HOW2_EXP/Timit/LM_train_files/'

# train_gen = DataLoader(files=glob.glob(train_path + "*"),
#                                 max_batch_label_len=20000,
#                                 max_batch_len=args.max_batch_len,
#                                 max_label_len=args.max_label_len,
#                                 Word_model=Word_model)   





# for i in range(10000):
#     B1 = train_gen.next()
#     print(len(B1.get('batch_names')) ,B1.get('smp_word_label').shape)
   
    #print("smp_feat,smp_label,smp_word_label,smp_trans_text,smp_feat_len,smp_label_len,smp_word_label_length,smp_word_text_length")
    #smp_feat,smp_label,smp_word_label,smp_trans_text,smp_feat_len,smp_label_len,smp_word_label_length,smp_word_text_length = B1
    #import pdb;pdb.set_trace()
    #print('smp_batch_no',i,"smp_feat,smp_label,smp_word_label,smp_trans_text,smp_feat_len,smp_label_len,smp_word_label_length,smp_word_text_length")
    #print(smp_feat.shape,smp_label.shape,smp_word_label.shape,smp_trans_text.shape,len(smp_feat_len),len(smp_label_len),len(smp_word_label_length),len(smp_word_text_length))
