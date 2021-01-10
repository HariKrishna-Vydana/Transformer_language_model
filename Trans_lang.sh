#! /bin/sh
#
#$ -q long.q@supergpu[123678],long.q@pc*,long.q@dellgpu*
#$ -l gpu=1,gpu_ram=7G,ram_free=7G,matylda3=0.5,'h=!*pcgpu4*'

#$ -o /mnt/matylda3/vydana/HOW2_EXP/LIBRISPEECH_RNNLM_V2/log/Trans_lang_6_512_1024_8_10000_accm8.log
#$ -e /mnt/matylda3/vydana/HOW2_EXP/LIBRISPEECH_RNNLM_V2/log/Trans_lang_6_512_1024_8_10000_accm8.log


PPATH="/mnt/matylda3/vydana/HOW2_EXP/LIBRISPEECH_RNNLM_V2"
cd "$PPATH"
export PYTHONUNBUFFERED=TRUE

###
gpu=1

tr_disp=1000
vl_disp=100

batch_size=1000
max_batch_len=1000
max_batch_label_len=5000

validate_interval=20000
max_val_examples=5567
max_feat_len=2000
max_label_len=500

##
decoder="Nothing"
dec_embd_vec_size=512
decoder_dmodel=1024
decoder_heads=8
decoder_dinner=4096
decoder_dropout=0.1
decoder_ff_dropout=0.1
decoder_layers=6
tie_dec_emb_weights=0
###
warmup_steps=10000
nepochs=200
step_num=1
learning_rate=1.0
lr_scale=1
clip_grad_norm=0
accm_grad=8
label_smoothing=0


####
early_stopping_checkpoints=8
early_stopping_patience=10
no_of_checkpoints=8
weight_noise_flag=1
early_stopping=1
reduce_learning_rate_flag=0

##
Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech_V2/models_train_text100/Librispeech_text100__bpe.model'
Char_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech_V2/models_train_text100/Librispeech_text100__bpe.model'
text_file='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech_V2/Librispeech_All_text_normalized'

train_path='/mnt/matylda3/vydana/HOW2_EXP/Librispeech_V2/LIBRISPEECH_LM_data/LM_train_files/'
dev_path='/mnt/matylda3/vydana/HOW2_EXP/Librispeech_V2/LIBRISPEECH_LM_data/LM_dev_files/'
pre_trained_weight="/mnt/matylda3/vydana/HOW2_EXP/LIBRISPEECH_RNNLM_V2/models/Trans_lang_6_512_1024_8_10000_accm8/model_epoch_8_sample_19999_81.3643059378147___75.62977127595381"

##

##
model_file="Trans_lang_6_512_1024_8_10000_accm8"
model_dir="$PPATH/models/$model_file"
weight_text_file="$PPATH/weight_files/$model_file"
Res_text_file="$PPATH/weight_files/$model_file"_Res

mkdir -pv $model_dir

output_file="$PPATH/log/$model_file".log
log_file="$PPATH/log/$model_file".log

if [[ ! -w $weight_text_file ]]; then touch $weight_text_file; fi
if [[ ! -w $Res_text_file ]]; then touch $Res_text_file; fi
echo "$model_dir"
echo "$weight_file"
echo "$Res_file"
#---------------------------------------------------------------------------------------------
stdbuf -o0  python /mnt/matylda3/vydana/HOW2_EXP/TRANS_LM_V1/Trans_LSTM_Training.py \
						--gpu $gpu \
						--tr_disp $tr_disp \
						--vl_disp $vl_disp \
						--batch_size $batch_size \
						--max_batch_len $max_batch_len \
                                                --max_batch_label_len $max_batch_label_len \
                                                --validate_interval $validate_interval \
						--max_val_examples $max_val_examples \
						--max_feat_len $max_feat_len \
						--max_label_len $max_label_len \
						--decoder $decoder \
						--dec_embd_vec_size $dec_embd_vec_size \
						--decoder_dmodel $decoder_dmodel \
						--decoder_heads $decoder_heads \
						--decoder_dinner $decoder_dinner \
						--decoder_dropout $decoder_dropout \
						--decoder_ff_dropout $decoder_ff_dropout \
						--decoder_layers $decoder_layers \
						--tie_dec_emb_weights $tie_dec_emb_weights \
						--warmup_steps $warmup_steps \
						--nepochs $nepochs \
						--step_num $step_num \
						--learning_rate $learning_rate \
						--lr_scale $lr_scale \
						--clip_grad_norm $clip_grad_norm \
						--accm_grad $accm_grad \
						--label_smoothing $label_smoothing \
						--early_stopping_checkpoints $early_stopping_checkpoints \
						--early_stopping_patience $early_stopping_patience \
						--no_of_checkpoints $no_of_checkpoints \
						--weight_noise_flag $weight_noise_flag \
						--early_stopping $early_stopping \
						--reduce_learning_rate_flag $reduce_learning_rate_flag \
						--Word_model_path $Word_model_path \
						--Char_model_path $Char_model_path \
						--text_file $text_file \
						--pre_trained_weight $pre_trained_weight \
						--model_dir $model_dir \
						--weight_text_file $weight_text_file \
						--Res_text_file $Res_text_file \
                                                --train_path $train_path \
                                                --dev_path $dev_path

#---------------------------------------------------------------------------------------------






