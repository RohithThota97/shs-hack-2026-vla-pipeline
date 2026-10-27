"""LoRA for SmolVLA in the curriculum trainer (`lora_rank` > 0), after the team's train_curriculum.py
(github.com/yugalrk/Robotics--VLA-training).

Each Linear layer in scope is wrapped: y = base(x) + (alpha / r) · B(A(dropout(x))), base frozen, B starts at zero (no
change at step 0). targets "vlm": the VLM (vision encoder, connector, language layers) gets adapters and its base
weights are frozen; the action expert and the projections train fully. "all": the expert gets adapters too.
Checkpoints are saved merged (plain Linear layers, a standard LeRobot checkpoint that serves like any other), plus
trainable_params.pt (adapters + fully trained parts) so a resumed run continues the adapters themselves.
"""
from __future__ import annotations

from typing import Any

import torch


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base, self.scale = base, alpha / r
        for p in base.parameters():
            p.requires_grad = False
        dev = base.weight.device
        self.lora_A = torch.nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.drop = torch.nn.Dropout(dropout)

    @property
    def weight(self) -> torch.Tensor:          # SmolVLA reads q_proj.weight.dtype
        return self.base.weight

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        A, B = self.lora_A.to(x.dtype), self.lora_B.to(x.dtype)     # bf16 VLM weights; autocast in training
        return y + (self.scale * (self.drop(x) @ A.t() @ B.t())).to(y.dtype)

    def delta(self) -> torch.Tensor:
        return (self.scale * self.lora_B @ self.lora_A).to(self.base.weight.dtype)


def scopes(policy: Any, targets: str = "vlm") -> list[torch.nn.Module]:
    """Modules whose Linear layers get LoRA (SmolVLA only)."""
    vwe = getattr(getattr(policy, "model", None), "vlm_with_expert", None)
    if vwe is None:
        raise ValueError("lora_rank is implemented for SmolVLA only")
    return [vwe.vlm] if targets == "vlm" else [vwe.vlm, vwe.lm_expert]


def add_lora(modules: list[torch.nn.Module], r: int, alpha: float | None = None, dropout: float = 0.0) -> int:
    """Freeze the modules and wrap their Linear layers (not lm_head) in LoRA. Returns the number of adapters."""
    n = 0
    for scope in modules:
        for p in scope.parameters():
            p.requires_grad = False
        for mod in list(scope.modules()):
            for name, child in list(mod.named_children()):
                if isinstance(child, torch.nn.Linear) and name != "lm_head":
                    setattr(mod, name, LoRALinear(child, r, alpha or 2 * r, dropout))
                    n += 1
    return n


def adapters(model: torch.nn.Module) -> list[tuple[torch.nn.Module, str, LoRALinear]]:
    return [(m, name, c) for m in model.modules() for name, c in m.named_children() if isinstance(c, LoRALinear)]


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


class merged:
    """Context: LoRA folded into plain Linear layers (a standard checkpoint for save_pretrained), restored after."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def __enter__(self) -> torch.nn.Module:
        self.saved = []
        with torch.no_grad():
            for m, name, lo in adapters(self.model):
                self.saved.append((m, name, lo, lo.base.weight.detach().cpu().clone()))
                lo.base.weight += lo.delta()
                setattr(m, name, lo.base)
        return self.model

    def __exit__(self, *exc: Any) -> None:
        with torch.no_grad():
            for m, name, lo, w in self.saved:
                lo.base.weight.copy_(w.to(lo.base.weight.device))
                setattr(m, name, lo)
