import argparse
import os

# Set endpoint before importing Hugging Face related libraries.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from utils import set_hf_endpoint, count_trainable_parameters
from data import gsm8k_dataset
from lora import LoRAConfig, inject_lora, lora_state_dict, mark_only_lora_trainable


def _format_sample(question: str, answer: str) -> str:
  return (
      "You are a helpful math tutor. Solve the problem step by step and end with '#### <answer>'.\n\n"
      f"Question: {question}\n"
      f"Answer: {answer}"
  )


def _tokenize_function(examples, tokenizer, max_length: int):
  texts = [_format_sample(q, a) for q, a in zip(examples["question"], examples["answer"])]
  outputs = tokenizer(
      texts,
      truncation=True,
      max_length=max_length,
      padding=False,
  )
  outputs["labels"] = outputs["input_ids"].copy()
  return outputs


def parse_args():
  parser = argparse.ArgumentParser(description="Manual LoRA fine-tuning for GSM8K")
  parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
  parser.add_argument("--output_dir", type=str, default="outputs/qwen2.5-7b-gsm8k-lora")
  parser.add_argument("--max_length", type=int, default=512)
  parser.add_argument("--epochs", type=int, default=2)
  parser.add_argument("--learning_rate", type=float, default=1e-4)
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--grad_accum", type=int, default=16)
  parser.add_argument("--max_train_samples", type=int, default=None)
  parser.add_argument("--max_steps", type=int, default=-1)
  parser.add_argument("--smoke_test", action="store_true")
  return parser.parse_args()


def main():
  args_cli = parse_args()
  set_hf_endpoint("https://hf-mirror.com")
  dataset = gsm8k_dataset()["train"]
  model_name = args_cli.model_name

  if args_cli.smoke_test:
    args_cli.max_train_samples = 32
    args_cli.max_steps = 5
    args_cli.output_dir = "outputs/smoke-qwen2.5-7b-lora"
    print("Smoke test mode: 32 samples, 5 training steps")

  if args_cli.max_train_samples is not None:
    dataset = dataset.select(range(min(args_cli.max_train_samples, len(dataset))))
    print(f"Using {len(dataset)} training samples")

  tokenizer = AutoTokenizer.from_pretrained(
      model_name,
      use_fast=False
    )
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
  )

  model.config.use_cache = False # 训练时建议关掉
  model.gradient_checkpointing_enable()

  lora_config = LoRAConfig(
      r=8,
      alpha=16,
      dropout=0.05,
  )
  replaced = inject_lora(model, lora_config)
  mark_only_lora_trainable(model)

  trainable, total, ratio = count_trainable_parameters(model)
  print(f"Injected LoRA into {replaced} linear layers")
  print(f"Trainable params: {trainable:,} / {total:,} ({ratio:.4f}%)")

  tokenized_train = dataset.map(
      lambda x: _tokenize_function(x, tokenizer, max_length=args_cli.max_length),
      batched=True,
      remove_columns=dataset.column_names,
      desc="Tokenizing train split",
  )

  args = TrainingArguments(
      output_dir=args_cli.output_dir,
      num_train_epochs=args_cli.epochs,
      learning_rate=args_cli.learning_rate,
      per_device_train_batch_size=args_cli.batch_size,
      gradient_accumulation_steps=args_cli.grad_accum,
      logging_steps=10,
      save_steps=200,
      save_total_limit=2,
      max_steps=args_cli.max_steps,
      fp16=torch.cuda.is_available(),
      bf16=False,
      report_to="none",
      remove_unused_columns=False,
  )

  trainer = Trainer(
      model=model,
      args=args,
      train_dataset=tokenized_train,
      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
  )
  trainer.train()

  trainer.save_model(args.output_dir)
  tokenizer.save_pretrained(args.output_dir)
  torch.save(lora_state_dict(model), f"{args.output_dir}/lora_only.bin")


if __name__ == "__main__":
  main()