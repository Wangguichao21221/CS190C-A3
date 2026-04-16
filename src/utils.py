import os
import torch.nn as nn


def set_hf_endpoint(endpoint):
	# Force override so child imports always read the expected endpoint.
	os.environ["HF_ENDPOINT"] = endpoint
	# Keep a second commonly used key for compatibility across versions.
	os.environ["HUGGINGFACE_HUB_ENDPOINT"] = endpoint


def count_trainable_parameters(model: nn.Module) -> tuple[int, int, float]:
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	ratio = (100.0 * trainable / total) if total else 0.0
	return trainable, total, ratio
