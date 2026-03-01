import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


class MoralFoundation(Dataset):
    """huggingface dataset wrapper for mbti text classification.
    """

    def __init__(
            self,
            encoder: SentenceTransformer,
            dataset_path: str = 'USC-MOLA-Lab/MFRC',
            split: str = 'train'
    ):
        full_data = load_dataset(dataset_path, split=split)

        self.data = full_data.shuffle(seed=42).select(
            range(int(len(full_data) * 0.1)))

        texts = [str(text) for text in self.data['text']]
        labels = self.data['annotation']

        self.embeddings = encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=256
        )
        # map string labels to integers
        raw_annotations = [str(a) if a is not None else "Non-Moral" for a in
                           self.data['annotation']]

        all_labels = []
        for a in raw_annotations:
            # split by comma if multi-label, strip whitespace
            parts = [p.strip() for p in a.split(',')]
            all_labels.extend(parts)

        self.unique_labels = sorted(list(set(all_labels)))
        self.num_classes = len(self.unique_labels)
        self.label_to_idx = {label: i for i, label in
                             enumerate(self.unique_labels)}
        self.idx_to_label = {i: label for label, i in
                             self.label_to_idx.items()}

        print(f"\n# USC-MOLA-Lab/MFRC MAPPING:\n{self.idx_to_label}")

        indices = [self.label_to_idx[
                       str(a).split(',')[0].strip()] if a is not None else
                   self.label_to_idx["Non-Moral"] for a in
                   self.data['annotation']]
        self.labels = torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param idx: sample index.
        """
        return self.embeddings[idx], self.labels[idx]