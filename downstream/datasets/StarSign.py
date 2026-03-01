import os
import torch
import pandas as pd
import kagglehub
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer


class StarSignDataset(Dataset):
    """Automated Kaggle-to-Torch pipeline for Horoscope data.

    Args:
        handle (str): Kaggle dataset handle (e.g., 'shahp7575/horoscopes').
        filename (str): The specific CSV file within the dataset.
        model_name (str): Transformer backbone for semantic embeddings.
    """

    def __init__(self,
                 handle: str = 'shahp7575/horoscopes',
                 filename: str = 'horoscopes.csv',
                 encoder: SentenceTransformer | torch.nn.Module = SentenceTransformer('all-MiniLM-L6-v2')):
        # get from kaggle
        cache_path = kagglehub.dataset_download(handle)
        full_path = os.path.join(cache_path, filename)
        df = pd.read_csv(full_path)

        texts = df['horoscope'].astype(str).tolist()
        self.signs = sorted(df['sign'].unique())
        self.sign_to_idx = {sign: i for i, sign in enumerate(self.signs)}
        labels = df['sign'].map(self.sign_to_idx).tolist()

        self.embeddings = encoder.encode(texts,
                                         convert_to_tensor=True,
                                         show_progress_bar=True)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]
