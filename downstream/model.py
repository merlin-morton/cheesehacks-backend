import torch
import torch.nn as nn

from model import SharedEncoderBinaryHeads
from huggingface_hub import hf_hub_download


class CoolProjectionHead(nn.Module):
    """
    maps the embedded space of moral vector space into some arbitrary
    downstream task space.
    """

    def __init__(self,
                 in_features: int = 768,
                 out_classes: int = 16,
                 checkpoint_path: str = None):
        super().__init__()

        # backbone
        if checkpoint_path is None:
            # download from hub if no local path provided
            downloaded_path = hf_hub_download(
                repo_id="Praneet-P/ethics-multihead-model",
                filename="checkpoints/shared_encoder_heads.pt"
            )
            checkpoint = torch.load(downloaded_path, map_location="cpu")

            self.encoder = SharedEncoderBinaryHeads(
                input_dim=checkpoint['input_dim'],
                latent_dim=checkpoint['latent_dim'],
                tasks=checkpoint['tasks']
            )
            self.encoder.load_state_dict(checkpoint["model_state"])
            in_features = checkpoint['latent_dim']
        else:
            self.encoder = SharedEncoderBinaryHeads(
                input_dim=in_features,
                latent_dim=in_features,
            )

        # freeze backbone
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # projection mlp
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_classes * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(out_classes * 2, out_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input embeddings [B, transformer_features]
        :returns: task logits [B, out_classes]
        """
        # we suppose x are preencoded into the embeddings
        features = self.encoder(x)
        return self.mlp(features)