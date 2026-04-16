import os

def gsm8k_dataset():
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    print(f"HUGGINGFACE_HUB_ENDPOINT={os.environ.get('HUGGINGFACE_HUB_ENDPOINT')}")
    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main")
    return dataset