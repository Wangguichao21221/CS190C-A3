import argparse
import os


import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from utils import set_hf_endpoint, count_trainable_parameters
from data import gsm8k_dataset
from lora import LoRAConfig, inject_lora, lora_state_dict, mark_only_lora_trainable


def _format_prompt(question: str) -> str:
  return (
      "You are a helpful math tutor. Solve the problem step by step and end with '#### <answer>'.\n\n"
      f"Question: {question}\n"
      "Answer: "
  )


def _format_sample(question: str, answer: str, eos_token: str) -> str:
  return f"{_format_prompt(question)}{answer}{eos_token}"


def _tokenize_function(examples, tokenizer, max_length: int):
  eos_token = tokenizer.eos_token or ""
  prompt_texts = [_format_prompt(q) for q in examples["question"]]
  texts = [_format_sample(q, a, eos_token) for q, a in zip(examples["question"], examples["answer"])]
  prompt_outputs = tokenizer(
      prompt_texts,
      truncation=True,
      max_length=max_length,
      padding=False,
  )
  outputs = tokenizer(
      texts,
      truncation=True,
      max_length=max_length,
      padding=False,
  )

  labels = []
  for prompt_ids, input_ids in zip(prompt_outputs["input_ids"], outputs["input_ids"]):
    prompt_length = min(len(prompt_ids), len(input_ids))
    label = [-100] * prompt_length + input_ids[prompt_length:]
    labels.append(label)

  outputs["labels"] = labels
  return outputs


class DataCollatorForCompletionOnlyLM:
  def __init__(self, tokenizer):
    self.tokenizer = tokenizer

  def __call__(self, features):
    labels = [torch.tensor(feature.pop("labels"), dtype=torch.long) for feature in features]
    batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
    batch["labels"] = pad_sequence(labels, batch_first=True, padding_value=-100)
    return batch


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
  parser.add_argument("--resume_from_checkpoint", type=str, default=None)
  parser.add_argument("--save_checkpoints", action="store_true")
  parser.add_argument("--smoke_test", action="store_true")
  parser.add_argument("--lora_r", type=int, default=8)
  parser.add_argument("--lora_alpha", type=int, default=16)
  parser.add_argument("--lora_dropout", type=float, default=0.05)
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

  tensorboard_log_dir = os.path.join('./logs', "qwen2.5-7b-gsm8k-lora")
  os.environ["TENSORBOARD_LOGGING_DIR"] = tensorboard_log_dir

  tokenizer = AutoTokenizer.from_pretrained(
      model_name,
      use_fast=False
    )
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
  )

  model.config.use_cache = False # 训练时建议关掉
  model.gradient_checkpointing_enable()

  lora_config = LoRAConfig(
      r=args_cli.lora_r,
      alpha=args_cli.lora_alpha,
      dropout=args_cli.lora_dropout,
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
      save_strategy="steps" if args_cli.save_checkpoints else "no",
      save_steps=200,
      save_total_limit=1 if args_cli.save_checkpoints else None,
      max_steps=args_cli.max_steps,
      fp16=torch.cuda.is_available(),
      bf16=False,
      report_to="tensorboard",
      remove_unused_columns=False,
  )

  trainer = Trainer(
      model=model,
      args=args,
      train_dataset=tokenized_train,
      data_collator=DataCollatorForCompletionOnlyLM(tokenizer),
  )
  trainer.train(resume_from_checkpoint=args_cli.resume_from_checkpoint)

  tokenizer.save_pretrained(args.output_dir)
  torch.save(lora_state_dict(model), f"{args.output_dir}/lora_only.bin")


if __name__ == "__main__":
  main()