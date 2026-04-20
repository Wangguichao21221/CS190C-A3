export OMP_NUM_THREADS=1
export HF_ENDPOINT="https://hf-mirror.com"
export HUGGINGFACE_HUB_ENDPOINT="https://hf-mirror.com"

python3 src/eval.py \
	--model_name "Qwen/Qwen2.5-7B" \
	--lora_path "outputs/qwen2.5-7b-gsm8k-lora/lora_only.bin" \
	--eval_file "gsm8k_val.jsonl" \
	--output_file "results.jsonl" \
	--max_new_tokens 512 \
	--batch_size 1 \
	--lora_r 16 \
	--lora_alpha 32 \
	--lora_dropout 0.05