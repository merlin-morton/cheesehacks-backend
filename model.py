import torch
import torch.nn as nn

class SharedEncoderBinaryHeads(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, tasks: list[str], dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({t: nn.Linear(latent_dim, 1) for t in tasks})

    def forward(self, emb: torch.Tensor, task: str) -> torch.Tensor:
        z = self.encoder(emb)          # [B,H]
        return self.heads[task](z)     # [B,1] logits

    def embedding_forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.encoder(emb)
