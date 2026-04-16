export HF_ENDPOINT="https://hf-mirror.com"
export HUGGINGFACE_HUB_ENDPOINT="https://hf-mirror.com"

python src/train.py \
	--model_name "Qwen/Qwen2.5-7B" \
	--output_dir "outputs/qwen2.5-7b-gsm8k-lora" \
	--max_length 512 \
	--epochs 2 \
	--learning_rate 1e-4 \
	--batch_size 1 \
	--grad_accum 16 \
	--max_steps -1
