from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
	r: int = 8
	alpha: int = 16
	dropout: float = 0.05
	target_modules: tuple[str, ...] = (
		"q_proj",
		"k_proj",
		"v_proj",
		"o_proj",
		"gate_proj",
		"up_proj",
		"down_proj",
	)


class LoRALinear(nn.Module):
	"""A linear layer with frozen base weights and trainable low-rank adapters."""

	def __init__(self, base_layer: nn.Linear, r: int, alpha: int, dropout: float) -> None:
		super().__init__()
		if not isinstance(base_layer, nn.Linear):
			raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)}")
		if r <= 0:
			raise ValueError("LoRA rank r must be > 0")

		self.base_layer = base_layer
		self.r = r
		self.alpha = alpha
		self.scaling = alpha / r
		self.dropout = nn.Dropout(dropout)

		in_features = base_layer.in_features
		out_features = base_layer.out_features

		self.lora_A = nn.Parameter(torch.empty(r, in_features))
		self.lora_B = nn.Parameter(torch.zeros(out_features, r))

		nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
		nn.init.zeros_(self.lora_B)

		for p in self.base_layer.parameters():
			p.requires_grad = False

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		base_out = self.base_layer(x)
		lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
		return base_out + lora_out.to(base_out.dtype) * self.scaling


def _iter_target_linears(model: nn.Module, target_modules: Iterable[str]):
	targets = tuple(target_modules)
	for module_name, module in model.named_modules():
		if isinstance(module, nn.Linear) and module_name.endswith(targets):
			yield module_name, module


def inject_lora(model: nn.Module, config: LoRAConfig) -> int:
	"""Replace target nn.Linear layers by LoRALinear layers."""
	replaced = 0
	for module_name, module in list(_iter_target_linears(model, config.target_modules)):
		if "." in module_name:
			parent_name, child_name = module_name.rsplit(".", 1)
			parent = model.get_submodule(parent_name)
		else:
			parent = model
			child_name = module_name
		setattr(parent, child_name, LoRALinear(module, config.r, config.alpha, config.dropout))
		replaced += 1
	return replaced


def mark_only_lora_trainable(model: nn.Module) -> None:
	for name, param in model.named_parameters():
		param.requires_grad = ("lora_A" in name) or ("lora_B" in name)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
	return {
		name: tensor.detach().cpu()
		for name, tensor in model.state_dict().items()
		if "lora_A" in name or "lora_B" in name
	}


