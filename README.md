# LoRA Fine-tuning Report for Qwen2.5-7B on GSM8K

## Final Result
- Validation set: `gsm8k_val.jsonl` (100 samples)
- Final accuracy: **0.810 (81/100)**
- Output file: `results.jsonl`

This meets the assignment requirement of **>= 75% accuracy**.

## LoRA Hyperparameters
- `r`: 16
- `alpha`: 32
- `dropout`: 0.05
- Target modules:
	- `q_proj`
	- `k_proj`
	- `v_proj`
	- `o_proj`
	- `gate_proj`
	- `up_proj`
	- `down_proj`

## Training Details
- Base model: `Qwen/Qwen2.5-7B`
- Dataset (train split): `openai/gsm8k` (`main`)
- Max sequence length: 512
- Epochs: 2
- Learning rate: `5e-5`
- Per-device batch size: 1
- Gradient accumulation steps: 16

## Hardware
- Device: `NVIDIA-H20 * 1`(from the AI clusters)

## Reproducibility

### Train
```bash
bash run_train.sh
```

### Evaluate
```bash
bash run_evaluate.sh
```
