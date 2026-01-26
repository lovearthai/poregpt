export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=748830e9b9acdf804bb0baad0eb82e6ca2592354
torchrun --nproc_per_node=1 --master_port 29507 -m poregpt.tokenizers.vqe_tokenizer.vqe_train --config config_pass21_c64k_ema.yaml
#python3  -m accelerate.launch  -m poregpt.tokenizers.vqe_tokenizer.vqe_train --config config_pass21_c64k_ema.yaml 
#accelerate launch /mnt/nas_syy/default/poregpt/poregpt/poregpt/tokenizers/vqe_tokenizer/vqe_train.py  --config config_pass21_c64k_ema.yaml 
