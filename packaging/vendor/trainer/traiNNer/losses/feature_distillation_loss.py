"""Feature distillation for BOLT (and other same-trunk archs).

Adds an auxiliary loss that pulls the student's per-stage LR-res feature maps
toward the frozen teacher's, via tiny learnable 1x1 channel adapters
(student_C -> teacher_C). The teacher and student consume the SAME LQ input,
so stage features share spatial size; only channel counts differ, which the
adapters reconcile.

Adapters are the ONLY distillation parameters (student_C*teacher_C per stage,
~69k params each for 180->384) and are TRAINED alongside the student but
DISCARDED at deployment — the exported BOLT-S is unchanged.

Used by sr_model only when train.distill_features is enabled and a teacher is
configured. Default-off, so normal training is unaffected.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FeatureDistiller"]


class FeatureDistiller(nn.Module):
    def __init__(
        self,
        n_stages: int,
        student_c: int,
        teacher_c: int,
        loss_type: str = "l1",
        stage_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            [nn.Conv2d(student_c, teacher_c, 1) for _ in range(n_stages)]
        )
        if loss_type not in ("l1", "l2", "cosine"):
            raise ValueError(f"unsupported distill_feature_loss: {loss_type}")
        self.loss_type = loss_type
        if stage_weights is None:
            stage_weights = [1.0] * n_stages
        assert len(stage_weights) == n_stages
        self.register_buffer(
            "stage_weights",
            torch.tensor(stage_weights, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        student_feats: list[torch.Tensor],
        teacher_feats: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert len(student_feats) == len(self.adapters), (
            f"student produced {len(student_feats)} stage feats, "
            f"distiller expects {len(self.adapters)}"
        )
        assert len(teacher_feats) == len(self.adapters), (
            f"teacher produced {len(teacher_feats)} stage feats, "
            f"distiller expects {len(self.adapters)}"
        )
        total = student_feats[0].new_zeros(())
        parts: dict[str, torch.Tensor] = {}
        for i, adapter in enumerate(self.adapters):
            proj = adapter(student_feats[i])
            tgt = teacher_feats[i].detach()
            if proj.shape[-2:] != tgt.shape[-2:]:
                proj = F.interpolate(
                    proj, size=tgt.shape[-2:], mode="bilinear", align_corners=False
                )
            if self.loss_type == "l1":
                li = (proj - tgt).abs().mean()
            elif self.loss_type == "l2":
                li = (proj - tgt).pow(2).mean()
            else:  # cosine
                a = F.normalize(proj, dim=1)
                b = F.normalize(tgt, dim=1)
                li = (1.0 - (a * b).sum(dim=1)).mean()
            total = total + self.stage_weights[i] * li
            parts[f"stage{i}"] = li.detach()
        total = total / len(self.adapters)
        return total, parts
