export OMP_NUM_THREADS=1
export HF_ENDPOINT="https://hf-mirror.com"
export HUGGINGFACE_HUB_ENDPOINT="https://hf-mirror.com"

python3 src/train.py \
	--model_name "Qwen/Qwen2.5-7B" \
	--output_dir "outputs/qwen2.5-7b-gsm8k-lora" \
	--max_length 512 \
	--epochs 2 \
	--learning_rate 5e-5 \
	--batch_size 1 \
	--grad_accum 16 \
	--max_steps -1 \
	--lora_r 16 \
	--lora_alpha 32 \
	--lora_dropout 0.05 \
	--save_checkpoints \

