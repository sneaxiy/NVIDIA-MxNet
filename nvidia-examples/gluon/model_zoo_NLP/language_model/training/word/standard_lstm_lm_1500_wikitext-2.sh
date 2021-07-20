NUM_GPUS=1
. ../../../set_env_vars.sh
set_env_vars 750

python ../../word_language_model.py --gpu $GPUS --emsize 1500 --nhid 1500 --nlayers 2 --lr 20 --epochs $NUM_EPOCHS --batch_size 20 --bptt 35 --dropout 0.65 --dropout_h 0 --dropout_i 0 --dropout_e 0 --weight_drop 0 --tied --wd 0 --alpha 0 --beta 0 --ntasgd --lr_update_interval 30 --lr_update_factor 0.1 --save standard_lstm_lm_1500_wikitext-2


