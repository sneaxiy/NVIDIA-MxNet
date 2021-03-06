NUM_GPUS=1
. ../../../set_env_vars.sh 
set_env_vars 12

MXNET_GPU_MEM_POOL_TYPE=Round
python ../../train_gnmt.py --src_lang en --tgt_lang vi --batch_size 128 --optimizer adam --lr 0.001 --lr_update_factor 0.5 --beam_size 10 --bucket_scheme exp --num_hidden 512 --save_dir gnmt_en_vi_l2_h512_beam10 --epochs $NUM_EPOCHS --gpu $GPUS
