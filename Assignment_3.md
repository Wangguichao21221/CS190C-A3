# Assignment 3: Hands-on Implementation of LoRA for Fine-tuning Qwen2.5-7B on GSM8K

**Deadline:** 2026-4-27 23:59

## Task  
In this assignment, you will implement Low-Rank Adaptation (LoRA) from scratch and apply it to fine-tune the [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) language model on the [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k). You can only use the training set to train the model. Then you will evaluate the performance of the fine-tuned model on the given validation set (named gsm8k_val.jsonl). Your goal is to achieve an **accuracy of at least 75%**.

## Evaluation and Output Format
- After training, evaluate your model on the given validation set (named gsm8k_val.jsonl). 
- Parse the final numerical answer from the model’s output. The answer is typically presented at the end in the format `#### {number}`. Extract `{number}` as the predicted result.
- Compare the parsed number with the ground-truth answer (provided in the dataset).
- Generate a JSONL file (`results.jsonl`) containing one JSON object per line with the following fields:
```json
{
  "question": "string",
  "ground_truth": "string",
  "model_output": "string",
  "parsed_answer": "string or null",
  "is_correct": true/false
}
```
## Accuracy Requirement
- Your implementation must achieve **≥75% accuracy** on the given validation set.
- Report your final accuracy in a `README.md` file along with:
  - LoRA hyperparameters (`r`, `alpha`, etc.)
  - Training details (`epochs`, `batch size`, `learning rate`, `hardware used`, etc.)

## Deliverables
- Your code with the `README.md` file
- Your `results.jsonl` file
- Submit above files to gradescope.

## Note
- You are allowed to use `transformers`, `datasets`, `torch`, etc., but LoRA must be implemented manually. Do not use high-level libraries like peft for LoRA injection.
- Parsing logic must be robust—handle cases where the model does not follow the expected output format.
