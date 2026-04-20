from __future__ import annotations

import argparse
import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora import LoRAConfig, inject_lora
from utils import set_hf_endpoint
from tqdm import tqdm

NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def parse_args():
  parser = argparse.ArgumentParser(description="Evaluate a LoRA-tuned model on GSM8K validation data")
  parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
  parser.add_argument("--lora_path", type=str, default="outputs/qwen2.5-7b-gsm8k-lora/lora_only.bin")
  parser.add_argument("--eval_file", type=str, default="gsm8k_val.jsonl")
  parser.add_argument("--output_file", type=str, default="results.jsonl")
  parser.add_argument("--max_new_tokens", type=int, default=256)
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--max_samples", type=int, default=None)
  parser.add_argument("--lora_r", type=int, default=8)
  parser.add_argument("--lora_alpha", type=int, default=16)
  parser.add_argument("--lora_dropout", type=float, default=0.05)
  return parser.parse_args()


def _build_prompt(question: str) -> str:
  return (
      "You are a helpful math tutor. Solve the problem step by step and you must end with '#### <answer>'. answer keep the answers short.\n\n"
      f"Question: {question}\n"
      "Answer:"
  )


def _load_validation_data(eval_file: str, max_samples: int | None = None) -> list[dict[str, str]]:
  records: list[dict[str, str]] = []
  with open(eval_file, "r", encoding="utf-8") as handle:
    for line in handle:
      if not line.strip():
        continue
      record = json.loads(line)
      records.append({
          "question": record["question"],
          "answer": record["answer"],
      })
      if max_samples is not None and len(records) >= max_samples:
        break
  return records


def _canonicalize_number(text: str) -> str | None:
  cleaned = text.strip().replace(",", "")
  cleaned = cleaned.rstrip(".")
  if not cleaned:
    return None
  if cleaned.startswith("+"):
    cleaned = cleaned[1:]
  try:
    value = Decimal(cleaned)
  except InvalidOperation:
    return None
  normalized = value.normalize()
  if normalized == normalized.to_integral():
    return str(normalized.to_integral())
  normalized_text = format(normalized, "f").rstrip("0").rstrip(".")
  if normalized_text in {"", "-0"}:
    return "0"
  return normalized_text


def _first_answer_span(text: str) -> str:
  """Keep only the first answer block and drop any continuation into the next problem."""
  if "####" not in text:
    return ""

  answer_text = text.split("####", 1)[1]

  cut_positions = [len(answer_text)]
  for marker in ("\nQuestion:", "\n\nQuestion:", "\nAnswer:", "\n\nAnswer:"):
    marker_pos = answer_text.find(marker)
    if marker_pos != -1:
      cut_positions.append(marker_pos)

  return answer_text[: min(cut_positions)].strip()


def _extract_parsed_answer(text: str) -> str | None:
  candidate_text = _first_answer_span(text)
  matches = NUMBER_PATTERN.findall(candidate_text)
  if not matches:
    return None
  return _canonicalize_number(matches[0])


def _extract_ground_truth(text: str) -> str | None:
  return _extract_parsed_answer(text)


def _resolve_lora_only_path(lora_path: str) -> Path:
  resolved_path = Path(lora_path)
  if resolved_path.is_dir():
    resolved_path = resolved_path / "lora_only.bin"
  if resolved_path.name != "lora_only.bin":
    raise ValueError(
        f"Expected a LoRA-only checkpoint file named lora_only.bin, got: {resolved_path}"
    )
  if not resolved_path.exists():
    raise FileNotFoundError(f"LoRA-only checkpoint not found: {resolved_path}")
  return resolved_path


def _load_lora_weights(model: torch.nn.Module, lora_path: str) -> None:
  resolved_path = _resolve_lora_only_path(lora_path)
  state_dict = torch.load(resolved_path, map_location="cpu")

  non_lora_keys = [
      key for key in state_dict.keys() if ("lora_A" not in key and "lora_B" not in key)
  ]
  if non_lora_keys:
    raise ValueError(
        "Expected a LoRA-only state dict, but found non-LoRA keys. "
        f"Examples: {non_lora_keys[:5]}"
    )

  missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
  if unexpected_keys:
    raise RuntimeError(f"Unexpected keys while loading LoRA weights: {unexpected_keys}")

  adapter_keys = [key for key in state_dict.keys() if "lora_A" in key or "lora_B" in key]
  print(f"Loaded LoRA-only checkpoint from {resolved_path}")
  print(f"Detected {len(adapter_keys)} LoRA tensors")
  if missing_keys:
    print(
        f"Model load completed with {len(missing_keys)} missing base keys, "
        "which is expected for a LoRA-only checkpoint"
    )


def _run_generation(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
  encoded = tokenizer(
      prompts,
      return_tensors="pt",
      padding=True,
      truncation=True,
  )
  model_device = next(model.parameters()).device
  encoded = {key: value.to(model_device) for key, value in encoded.items()}
  prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()

  with torch.inference_mode():
    generated = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

  outputs: list[str] = []
  for sequence, prompt_length in zip(generated, prompt_lengths):
    continuation = sequence[int(prompt_length):]
    outputs.append(tokenizer.decode(continuation, skip_special_tokens=True))
  return outputs


def main():
  args_cli = parse_args()
  set_hf_endpoint("https://hf-mirror.com")

  records = _load_validation_data(args_cli.eval_file, args_cli.max_samples)
  if not records:
    raise ValueError(f"No validation records found in {args_cli.eval_file}")

  use_cuda = torch.cuda.is_available()
  print(f"Using device: {'cuda' if use_cuda else 'cpu'}")
  torch_dtype = torch.float16 if use_cuda else torch.float32
  device = "cuda" if use_cuda else "cpu"

  tokenizer = AutoTokenizer.from_pretrained(args_cli.model_name, use_fast=False)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "left"

  model = AutoModelForCausalLM.from_pretrained(
      args_cli.model_name,
      torch_dtype=torch_dtype,
  )
  model.config.use_cache = True

  inject_lora(
      model,
      LoRAConfig(
          r=args_cli.lora_r,
          alpha=args_cli.lora_alpha,
          dropout=args_cli.lora_dropout,
      ),
  )
  _load_lora_weights(model, args_cli.lora_path)
  model = model.to(device)
  model.eval()

  output_path = Path(args_cli.output_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  total = 0
  correct = 0
  print (f"Evaluating {len(records)} samples...")
  with open(output_path, "w", encoding="utf-8") as writer:
    for start_index in tqdm(range(0, len(records), args_cli.batch_size)):
      batch = records[start_index:start_index + args_cli.batch_size]
      prompts = [_build_prompt(item["question"]) for item in batch]
      model_outputs = _run_generation(model, tokenizer, prompts, args_cli.max_new_tokens)

      for item, model_output in zip(batch, model_outputs):
        ground_truth = item["answer"]
        parsed_ground_truth = _extract_ground_truth(ground_truth)
        parsed_answer = _extract_parsed_answer(model_output)
        is_correct = parsed_answer is not None and parsed_ground_truth is not None and parsed_answer == parsed_ground_truth

        result = {
            "question": item["question"],
            "ground_truth": ground_truth,
            "model_output": model_output,
            "parsed_answer": parsed_answer,
            "is_correct": is_correct,
        }
        for key, value in result.items():
          print(f"{key}: {value}")
        print("-" * 40)
        writer.write(json.dumps(result, ensure_ascii=False) + "\n")

        total += 1
        if is_correct:
          correct += 1

  accuracy = (correct / total) if total else 0.0
  print(f"Evaluated {total} samples")
  print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
  print(f"Results written to {output_path}")


if __name__ == "__main__":
  main()