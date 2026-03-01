import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


class Briggs(Dataset):
    """huggingface dataset wrapper for mbti text classification.
    """

    def __init__(
            self,
            encoder: SentenceTransformer,
            dataset_path: str = 'Shunian/kaggle-mbti-cleaned',
            split: str = 'train'
    ):
        self.data = load_dataset(dataset_path, split=split)

        texts = [str(text) for text in self.data['text']]
        labels = self.data['label']

        self.embeddings = encoder.encode(texts, convert_to_tensor=True)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param idx: sample index.
        """
        return self.embeddings[idx], self.labels[idx]