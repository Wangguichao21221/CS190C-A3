import json
with open("results.jsonl") as f:
  for i, line in enumerate(f):
    record = json.loads(line)
    if record["is_correct"] is False:
      ground_truth = record["ground_truth"]
      model_output = record["model_output"]
      parsed_answer = record["parsed_answer"]
      print(f"Sample {i}:")
      print(f"question: {record['question']}")
      print(f"ground truth: {ground_truth}")
      print(f"model output: {model_output}")
      print(f"parsed result: {parsed_answer}")
      print("-" * 80)