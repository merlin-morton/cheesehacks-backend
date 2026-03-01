import torch
from downstream.model import CoolProjectionHead
from huggingface_hub import hf_hub_download

from model import SharedEncoderBinaryHeads


def load_mlp_model(checkpoint_path: str = None) -> SharedEncoderBinaryHeads:
    """Loads the MLP model from a checkpoint path.

    Args:
        checkpoint_path: Path to the saved weights.

    Returns:
        The loaded SharedEncoderBinaryHeads model.
    """
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = SharedEncoderBinaryHeads(
            input_dim=checkpoint['input_dim'],
            latent_dim=checkpoint['latent_dim'],
            tasks=checkpoint['tasks']
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)

    model.eval()
    return model


model_path = hf_hub_download(
    repo_id="Praneet-P/ethics-multihead-model",
    filename="checkpoints/shared_encoder_heads.pt"
)
embedding_model = load_mlp_model(model_path)


def load_classification_model(dataset_name: str) -> CoolProjectionHead:
    """Loads a specific downstream classification head.

    Args:
        dataset_name: Name of the dataset/head to load.

    Returns:
        The loaded CoolProjectionHead model.
    """
    model_path = hf_hub_download(
        repo_id="Praneet-P/ethics-multihead-model",
        filename=f"checkpoints/downstream/{dataset_name}.pt"
    )
    ckpt = torch.load(model_path, map_location="cpu")

    model = CoolProjectionHead(
        encoder=embedding_model,
        in_features=ckpt['in_features'],
        out_classes=ckpt['out_classes']
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    return model


@torch.no_grad()
def predict_class(
        model: torch.nn.Module,
        personality_vector: torch.Tensor,
        labels: list[str]
) -> str:
    """Runs inference and maps logit argmax to string label.

    Args:
        model: The classification head.
        personality_vector: The encoded representation.
        labels: List of string labels mapping to output indices.

    Returns:
        The predicted class string.
    """
    logits = model(personality_vector)
    label_idx = torch.argmax(logits, dim=-1).item()
    return labels[label_idx]


# global models
bigfive_model = load_classification_model('bigfive')
briggs_model = load_classification_model('briggs')
moralfoundation_model = load_classification_model('moralfoundation')
politicalleaning_model = load_classification_model('politicalleaning')
starsign_model = load_classification_model('starsign')

# mappings
BRIGGS_LABELS = [
    'ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP',
    'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'
]

MORAL_FOUNDATION_LABELS = [
    'Authority',
    'Care'
    'Equality',
    'Loyalty',
    'Non-Moral',
    'Proportionality',
    'Purity',
    'Thin Morality'
]


def _classify_pv(personality_vector: torch.Tensor) -> dict:
    briggs_pred = predict_class(briggs_model, personality_vector,
                                BRIGGS_LABELS)
    moral_pred = predict_class(moralfoundation_model, personality_vector,
                               MORAL_FOUNDATION_LABELS)

    return {
        "briggs": briggs_pred,
        "moral_foundation": moral_pred
    }
