import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb 


from torch.autograd import Variable

#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
def get_decoder_non_pad_mask(padded_input, pad_idx=0):
    """padding position is set to 0, either use input_lengths or pad_idx
    """
    # padded_input: N x T
    assert padded_input.dim() == 2
    non_pad_mask = padded_input.ne(pad_idx).float()
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
def get_encoder_non_pad_mask(padded_input, input_lengths):
    """padding position is set to 0, either use input_lengths or pad_idx
    """
    # padded_input: N x T x ..
    #print(padded_input.size())
    N = padded_input.size(0)
    #print(padded_input.size()[:-1])
    non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # N x T
    #print(non_pad_mask.size())
    for i in range(N):
        non_pad_mask[i, input_lengths[i]:] = 0
        #print(input_lengths[i])
    return non_pad_mask.unsqueeze(-1)
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    """mask position is set to 1"""
    # N x Ti x 1
    #print(padded_input.size())
    #print(input_lengths)
    non_pad_mask = get_encoder_non_pad_mask(padded_input, input_lengths=input_lengths)
    # N x Ti, lt(1) like not operation
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    #print(pad_mask)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    #print(attn_mask.size())
    #print(attn_mask)
    return attn_mask
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
def get_attn_pad_mask_encoder(padded_input, input_lengths, expand_length,win):
    """mask position is set to 1"""
    # N x Ti x 1
    #print(padded_input.size())
    #print(input_lengths)
    non_pad_mask = get_encoder_non_pad_mask(padded_input, input_lengths=input_lengths)
    # N x Ti, lt(1) like not operation
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    #print(pad_mask)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    #-------------------------------------------
    #-------------------------------------------
    return attn_mask
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------

def get_subsequent_mask(seq,win):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = torch.tril(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8))
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    #-------------------------------------------
    ##-------------------------------------------
    return subsequent_mask

#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------

MIN_VALUE = float(np.finfo(np.float32).min)
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        
        attn = torch.matmul(q, k.transpose(-2,-1))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask==0, MIN_VALUE)
        attn = self.softmax(attn)

        ##no self attn_dropout
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),attn_dropout=dropout)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        #pdb.set_trace()
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        #================================
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        #================================

        q = self.w_qs(q).view(sz_b, -1, n_head, d_k)
        k = self.w_ks(k).view(sz_b, -1, n_head, d_k)
        v = self.w_vs(v).view(sz_b, -1, n_head, d_v)
        #================================
        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)
        #================================


        #print(mask.size(),attn.size())
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1,n_head,1, 1)
            #print(mask.size(),attn.size())
            #mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        #================================    
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.transpose(1,2).contiguous().view(sz_b,-1, n_head*d_v)
        output = self.fc(output)
        return output, attn

#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        #self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        #self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        #self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        x=dec_input
        #----------------------------------
        #----------------------------------
        #if slf_attn_mask:
        #print(slf_attn_mask)
        #print("slf_attn_mask is none somethong is wrong")
        


        nx=self.norm1(x)
        dec_output, dec_slf_attn = self.slf_attn(nx, nx,nx, mask=slf_attn_mask)
        x=x+self.dropout(dec_output)

        #----------------------------------
        #----------------------------------
        #----------------------------------
        #----------------------------------
        nx=self.norm3(x)
        dec_output = self.pos_ffn(nx)
        dec_output=x+self.dropout(dec_output)
        #----------------------------------
        #----------------------------------

        dec_enc_attn=dec_slf_attn
        return dec_output, dec_slf_attn, dec_enc_attn
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,mask_id,IGNORE_ID,sos_id, eos_id,
            n_tgt_vocab, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner,dec_win,dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            pe_maxlen=5000):
        super(Decoder, self).__init__()
        # parameters
        self.sos_id = sos_id  # Start of Sentence
        self.eos_id = eos_id  # End of Sentence
        self.n_tgt_vocab = n_tgt_vocab
        self.d_word_vec = d_word_vec
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.tgt_emb_prj_weight_sharing = tgt_emb_prj_weight_sharing
        self.pe_maxlen = pe_maxlen
        self.dec_win=dec_win
        self.mask_id=mask_id

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec)
        self.x_scale = math.sqrt(d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)


        self.output_norm=nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab)

        if self.tgt_emb_prj_weight_sharing:
                self.tgt_word_emb.weight=self.tgt_word_prj.weight


        self.IGNORE_ID=IGNORE_ID
    def preprocess(self, padded_input,teacher_force_rate=1):
        """Generate decoder input and output label from padded_input
        Add <sos> to decoder input, and add <eos> to decoder output label
        """
        #print(padded_input)
        #--------------------------------------------------------
        #--------------------------------------------------------

        ys = [y[y != self.IGNORE_ID] for y in padded_input]  # parse padded ys
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos_id])
        sos = ys[0].new([self.sos_id])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys ]
       
        #--------------------------------------------------------       
        #--------------------------------------------------------
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        
        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos_id)
        
        ys_out_pad = pad_list(ys_out, self.IGNORE_ID) ####original
        

        #ys_out_pad = pad_list(ys_out, self.eos_id)######modified
        assert ys_in_pad.size() == ys_out_pad.size()
        return ys_in_pad, ys_out_pad

    def forward(self, padded_input, encoder_padded_outputs,  encoder_input_lengths,teacher_force_rate,return_attns=False):
        """ Args: --------> padded_input: N x To --------> encoder_padded_outputs: N x Ti x H -------->Returns: """

        #####################################################################
        #just assumes the that batches are arraged as per in put lengths and 
        #masks not required when the inputs are af almost asame size
        input_lengths=torch.tensor([encoder_padded_outputs.size(1)]).repeat(encoder_padded_outputs.size(0))
        cuda_flag=encoder_padded_outputs.is_cuda
        device_flag=torch.device("cuda") if cuda_flag else torch.device("cpu")

        encoder_input_lengths=input_lengths.cuda() if cuda_flag else input_lengths
        #===================================================================
        #===================================================================
        dec_slf_attn_list, dec_enc_attn_list = [], []
        # Get Deocder Input and Output
        ys_in_pad, ys_out_pad = self.preprocess(padded_input,teacher_force_rate)

       # Prepare masks
        slf_attn_mask_subseq = get_subsequent_mask(ys_in_pad,self.dec_win)
        #import pdb;pdb.set_trace()
        #print(slf_attn_mask_subseq)
        output_length = ys_in_pad.size(1)

        non_pad_mask=None;
        dec_enc_attn_mask=None
        #slf_attn_mask_subseq=None
        slf_attn_mask_keypad=None;
        slf_attn_mask=slf_attn_mask_subseq
        embd_output=self.tgt_word_emb(ys_in_pad)
        #===================================================================
        #===================================================================
        #===================================================================
        #if teacher_force_rate<1:
        #    mixup_list=torch.randint(low=1,high=embd_output.size(1),size=(int(embd_output.size(1)*teacher_force_rate),),device=device_flag)

        #    mask_index=torch.tensor([self.mask_id])            
        #    mask_index=mask_index.cuda() if cuda_flag else mask_index
        #    mask_embeding=self.tgt_word_emb(mask_index)
        #    for qqq in range(mixup_list.size(0)):
        #        embd_output[:,mixup_list[qqq],:]=mask_embeding.unsqueeze(0)
        #===================================================================
        #===================================================================
        #===================================================================
        dec_output_list=[]
        #dec_output=self.positional_encoding(embd_output)

        ###no pos_embeding as said in ACHEN paper
        dec_output=embd_output
        #***********
        ###for language model
        encoder_padded_outputs=dec_output
        #************

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, encoder_padded_outputs,non_pad_mask=non_pad_mask,slf_attn_mask=slf_attn_mask,dec_enc_attn_mask=dec_enc_attn_mask)
            dec_output_list.append(dec_output)
        #-----------------------------------------
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]
        #-----------------------------------------
        # before softmax
        dec_output=self.output_norm(dec_output)
        seq_logit = self.tgt_word_prj(dec_output)
        #-----------------------------------------
        # Return
        pred, gold = seq_logit, ys_out_pad
        if return_attns:
            return pred, gold, dec_slf_attn_list, dec_enc_attn_list
        return pred, gold,dec_output_list

#==========================================================================
#==========================================================================
#==========================================================================
#==========================================================================
#==========================================================================
    def prediction_from_trained_model(self,ys,encoder_outputs,scores_list):
                #import pdb;pdb.set_trace()
                non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1) # 1xix1
                slf_attn_mask = get_subsequent_mask(ys,self.dec_win)
                #slf_attn_mask = None
                # -- Forward

                #pdb.set_trace()
                #dec_output=self.positional_encoding(ys)
                #dec_output=self.positional_encoding(self.tgt_word_emb(ys))


                ###no pos_embeding as said in ACHEN paper
                dec_output=self.tgt_word_emb(ys)

                #--------------------------------------------
                encoder_outputs=dec_output
                for dec_layer in self.layer_stack:
                    dec_output, _, _ = dec_layer(dec_output, encoder_outputs,non_pad_mask=None,slf_attn_mask=slf_attn_mask,dec_enc_attn_mask=None)

                dec_output = self.output_norm(dec_output)
                seq_logit = self.tgt_word_prj(dec_output[:,-1])

                scores_list.append(seq_logit.unsqueeze(1))
                local_scores = F.log_softmax(seq_logit, dim=1)
                #pdb.set_trace()
                scores = F.softmax(seq_logit, dim=1)

                present_label=torch.argmax(scores, dim=1)
                return local_scores,scores_list,present_label,scores
#==========================================================================
    def prediction_from_trained_model_from_another_AM(self,ys,encoder_outputs,scores_list):
                #import pdb
                #pdb.set_trace()
                # pred,gold,dec_output_list=self.decoder.forward(padded_input,padded_input,encoder_input_lengths,teacher_force_rate)
                # decoder_LM_represenations=dec_output_list[-1]
                # Decode_Layer_add=False
                
                non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1) # 1xix1
                slf_attn_mask = get_subsequent_mask(ys,self.dec_win)
                #slf_attn_mask = None
                # -- Forward
                #pdb.set_trace()
                #dec_output=self.positional_encoding(ys)
                
                #dec_output=self.positional_encoding(self.tgt_word_emb(ys))

                #-----------------------------------------------
                ###no pos_embeding as said in ACHEN paper
                dec_output=self.tgt_word_emb(ys)
                #------------------------------------------------

                encoder_outputs=dec_output
                dec_output_list=[]
                for dec_layer in self.layer_stack:
                    dec_output, _, _ = dec_layer(dec_output, encoder_outputs,non_pad_mask=None,slf_attn_mask=slf_attn_mask,dec_enc_attn_mask=None)
                    dec_output_list.append(dec_output)

                #print(dec_output.size(),dec_output[:, -1].size())
                dec_output = self.output_norm(dec_output)
                seq_logit = self.tgt_word_prj(dec_output)

                scores_list.append(seq_logit.unsqueeze(1))
                local_scores = F.log_softmax(seq_logit, dim=1)
                #pdb.set_trace
                scores = F.softmax(seq_logit, dim=2)

                present_label=torch.argmax(scores, dim=2)
                return local_scores,scores_list,present_label,scores,dec_output_list
#==========================================================================
  
#===================================================================================================================
#==================================================================================================================
#==========================================================================
   
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------

class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """
    def __init__(self, decoder):
        super(Transformer, self).__init__()
        self.decoder = decoder
        #----------------------------------

    def forward(self, padded_input, input_lengths,padded_target,teacher_force_rate):
        """ Args: ------> padded_input: N x Ti x D ---------> input_lengths: N -------> padded_targets: N x To  """

        encoder_padded_outputs=padded_target
        # pred is score before softmax
        pred, gold,*_ = self.decoder(padded_target, encoder_padded_outputs,input_lengths,teacher_force_rate)
        return pred, gold

    def recognize(self, input, input_length,beam,gamma=1):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args: ----> input: T x D ----> char_list: list of characters -----> args: args.beam ------> Returns: -----> nbest_hyps: """

        char_list, args=0,0
        conv_padded_input=self.conv_layers(input)
        encoder_outputs, *_ = self.encoder(conv_padded_input, input_length)
        yg,nbest_hyps,scoring_list = self.decoder.recognize_beam_autoreg(encoder_outputs,char_list,args,beam,gamma)
        return nbest_hyps,0
    #=============================================================================================================
    #=============================================================================================================
    #==============================================================================
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
def cal_performance(pred, gold,IGNORE_ID,normalize_length=False,smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    Args:
        pred: N x T x C, score before softmax
        gold: N x T
    """
    #IGNORE_ID=-1
    pred = pred.view(-1, pred.size(2))
    gold = gold.contiguous().view(-1)


    loss = cal_loss(pred, gold,IGNORE_ID,normalize_length,smoothing)

    pred = pred.max(1)[1]
    non_pad_mask = gold.ne(IGNORE_ID)

    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    n_correct=n_correct/float(non_pad_mask.sum())
 
    n_correct=1.0-n_correct
    return loss, n_correct
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
#===============================================
#===============================================
def cal_loss(pred, gold,IGNORE_ID,normalize_length,smoothing):
    """Calculate cross entropy loss, apply label smoothing if needed. """
    normalize_length=True
    loss = F.cross_entropy(pred, gold,ignore_index=IGNORE_ID,reduction='mean')
    return loss

#====================================================
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
     """Implement the positional encoding (PE) function.

#     PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
#     PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
#     """
     def __init__(self, d_model, max_len=5000):
         super(PositionalEncoding, self).__init__()
         # Compute the positional encodings once in log space.
         pe = torch.zeros(max_len, d_model, requires_grad=False)
         self.dropout=nn.Dropout(0.1)

         position = torch.arange(0, max_len).unsqueeze(1).float()
         div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                              -(math.log(10000.0) / d_model))
         #div_term=10000 ** (torch.arange(0, d_model, 2)/d_model)
         #div_term=div_term.float()
         pe[:, 0::2] = torch.sin(position * div_term)
         pe[:, 1::2] = torch.cos(position * div_term)

         self.x_scale = math.sqrt(d_model)
         pe = pe.unsqueeze(0)
         self.register_buffer('pe', pe)
     def forward(self, input):
         """Args: input: N x T x D """

         length = input.size(1)
         PE=Variable(self.pe[:, :length],requires_grad=False)
         return self.dropout(input + PE)
#=============================================================================================================

#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    """Implements position-wise feedforward sublayer. FFN(x) = max(0, xW1 + b1)W2 + b2  """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = F.relu(self.w_1(x))
        output = self.dropout(output)
        output = self.w_2(output)
        return output

##############################################################################################################
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
class TransformerOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, k, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.k = k
        
        self.init_lr = d_model ** (-0.5)
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.reduction_factor=1
    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        lr = self.k * self.init_lr * min(self.step_num ** (-0.5), self.step_num * (self.warmup_steps ** (-1.5)))
             
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


#=============================================================================================================
#---------------------------------------------------------------------------------------------------------------
#===============================================================================================================

def pad_list(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad



def process_dict(dict_path):
    with open(dict_path, 'rb') as f:
        dictionary = f.readlines()
    char_list = [entry.decode('utf-8').split(' ')[0]
                 for entry in dictionary]
    sos_id = char_list.index('<sos>')
    eos_id = char_list.index('<eos>')
    return char_list, sos_id, eos_id


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    char_list, sos_id, eos_id = process_dict(path)
    print(char_list, sos_id, eos_id)
# * ------------------ recognition related ------------------ *

def parse_hypothesis(hyp, char_list):
    """Function to parse hypothesis

    :param list hyp: recognition hypothesis
    :param list char_list: list of characters
    :return: recognition text strinig
    :return: recognition token strinig
    :return: recognition tokenid string
    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp['yseq'][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp['score'])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace('<space>', ' ')

    return text, token, tokenid, score


def add_results_to_json(js, nbest_hyps, char_list):
    """Function to add N-best results to json

    :param dict js: groundtruth utterance dict
    :param list nbest_hyps: list of hypothesis
    :param list char_list: list of characters
    :return: N-best results added utterance dict
    """
    # copy old json info
    new_js = dict()
    new_js['utt2spk'] = js['utt2spk']
    new_js['output'] = []

    for n, hyp in enumerate(nbest_hyps, 1):
        # parse hypothesis
        rec_text, rec_token, rec_tokenid, score = parse_hypothesis(
            hyp, char_list)

        # copy ground-truth
        out_dic = dict(js['output'][0].items())

        # update name
        out_dic['name'] += '[%d]' % n

        # add recognition results
        out_dic['rec_text'] = rec_text
        out_dic['rec_token'] = rec_token
        out_dic['rec_tokenid'] = rec_tokenid
        out_dic['score'] = score

        # add to list of N-best result dicts
        new_js['output'].append(out_dic)

        # show 1-best result
        if n == 1:
            print('groundtruth: %s' % out_dic['text'])
            print('prediction : %s' % out_dic['rec_text'])

    return new_js


# -- Transformer Related --
#import torch
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
#============================================================================================
def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """padding position is set to 0, either use input_lengths or pad_idx
    """
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        # padded_input: N x T x ..
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # N x T
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 0
    if pad_idx is not None:
        # padded_input: N x T
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
#=======================================================================================
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
# Another implementation
class PositionwiseFeedForwardUseConv(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForwardUseConv, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
#=============================================================================================================
#-------------------------------------------------------------------------------------------------------------
