import torch
import torch.nn as nn


class CoolProjectionHead(nn.Module):
    """
    Maps the embedded space of moral vector space into some arbitrary
    downstream task space.

    Args:
        in_features: dimension of the moral manifold.
        out_classes: number of target classes
    """

    def __init__(self, in_features: int = 384, out_classes: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_classes * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_classes * 2, out_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: aggregated normative ethical vector. shape: [B, 5].

        Returns:
            raw class logits. shape: [B, out_classes].
        """
        return self.mlp(x)