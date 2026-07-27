"""Lightweight illumination-aware MoE stem for single-image correction."""
import torch
import torch.nn as nn

from .conv import Conv


class LightMoEStem(nn.Module):
    """Region-aware light correction stem with dark/over/normal experts."""

    def __init__(self, c1, hidden=32, num_experts=3):
        super().__init__()
        self.c1 = c1
        self.c2 = c1
        self.num_experts = max(int(num_experts), 3)
        hidden = max(int(hidden), 16)

        self.skip = nn.Identity()
        self.stem = Conv(c1, hidden, 3, 1)
        self.shared = nn.Sequential(Conv(hidden, hidden, 3, 1), Conv(hidden, hidden, 3, 1))
        self.experts = nn.ModuleList(
            nn.Sequential(Conv(hidden, hidden, 3, 1), Conv(hidden, hidden, 3, 1)) for _ in range(self.num_experts)
        )
        # self.experts = nn.ModuleList(
        #     nn.Sequential(Conv(hidden, hidden, 3, 1), Conv(hidden, hidden, 3, 1),
        #                   Conv(hidden, hidden, 3, 1), Conv(hidden, hidden, 3, 1)) for _ in range(self.num_experts)
        # )
        router_hidden = max(hidden // 2, self.num_experts)
        self.router = nn.Sequential(Conv(hidden, router_hidden, 1, 1), nn.Conv2d(router_hidden, self.num_experts + 1, 1))
        self.out = nn.Sequential(Conv(hidden, hidden, 3, 1), nn.Conv2d(hidden, c1, 1))

        self.last_corrected = None
        self.last_route_probs = None
        self.last_confidence = None
        self.last_mixed_feat = None
        self.last_delta = None
        self.last_expert_feats = None
        self.stack_expert_feats_in_eval = True

    def clear_cache(self):
        """Clear transient forward caches that should not participate in deepcopy/serialization."""
        self.last_corrected = None
        self.last_route_probs = None
        self.last_confidence = None
        self.last_mixed_feat = None
        self.last_delta = None
        self.last_expert_feats = None

    def __getstate__(self):
        """Drop non-leaf cached tensors so EMA deepcopy can clone the module safely."""
        state = self.__dict__.copy()
        for key in (
            "last_corrected",
            "last_route_probs",
            "last_confidence",
            "last_mixed_feat",
            "last_delta",
            "last_expert_feats",
        ):
            state[key] = None
        return state

    def _mix(self, x, return_experts=False):
        feat = self.stem(x)
        router_out = self.router(feat)
        route_logits = router_out[:, : self.num_experts]
        confidence = torch.sigmoid(router_out[:, self.num_experts : self.num_experts + 1])
        route_probs = route_logits.softmax(dim=1)

        mixed_feat = self.shared(feat)
        expert_feats = []
        for i, expert in enumerate(self.experts):
            expert_feat = expert(feat)
            expert_feats.append(expert_feat)
            mixed_feat = mixed_feat + expert_feat * route_probs[:, i : i + 1]

        delta = self.out(mixed_feat)
        corrected = torch.clamp(self.skip(x) + confidence * delta, 0.0, 1.0)
        if return_experts:
            return corrected, route_probs, confidence, mixed_feat, delta, torch.stack(expert_feats, dim=1)
        return corrected, route_probs, confidence, mixed_feat, delta

    def forward(self, x):
        return_experts = self.training or getattr(self, "stack_expert_feats_in_eval", True)
        if return_experts:
            corrected, route_probs, confidence, mixed_feat, delta, expert_feats = self._mix(x, return_experts=True)
            self.last_expert_feats = expert_feats
        else:
            corrected, route_probs, confidence, mixed_feat, delta = self._mix(x, return_experts=False)
            self.last_expert_feats = None
        self.last_corrected = corrected
        self.last_route_probs = route_probs
        self.last_confidence = confidence
        self.last_mixed_feat = mixed_feat
        self.last_delta = delta
        return corrected
