import torch

import numpy as np

from data.encode import get_encoding

from model.positionalencoding import PositionalEncoding


class TransformerModel(torch.nn.Module):
    def __init__(
        self, embedding_size, head_size, hidden_size, layers, dropout=0.5
    ) -> None:
        super(TransformerModel, self).__init__()

        self.encoding = get_encoding()

        self.embedding_size = embedding_size
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.layers = layers

        self.vocabulary_sizes = np.array(self.encoding["n_tokens"]) + 1
        self.total_embedding_size = embedding_size * len(self.encoding["dimensions"])

        self.mask_tokens = torch.tensor(self.vocabulary_sizes - 1)

        self.embedding = torch.nn.ModuleList(
            [
                torch.nn.Embedding(vocabulary_size, embedding_size)
                for vocabulary_size in self.vocabulary_sizes
            ]
        )
        self.positional_encoding = PositionalEncoding(
            self.total_embedding_size, dropout=dropout
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                self.total_embedding_size,
                self.head_size,
                self.hidden_size,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=self.layers,
        )

        self.reconstruction_heads = self._build_classification_heads()

    def _build_classification_heads(self):
        return torch.nn.ModuleList(
            [
                torch.nn.Linear(self.total_embedding_size, vocabulary_size)
                for vocabulary_size in self.vocabulary_sizes
            ]
        )

    def _embed(self, x):
        return torch.cat(
            [embedding(x[:, :, i]) for i, embedding in enumerate(self.embedding)],
            dim=-1,
        )

    def _reconstruct(self, x):
        return torch.cat([head(x) for head in self.reconstruction_heads], dim=-1)

    def _random_mask(self, x):
        mask_positions = (torch.rand(x.shape) < 0.15) * torch.stack(
            [torch.zeros(x.shape[:2])]
            + [x[:, :, 0] == self.encoding["type_code_map"]["note"]] * (x.shape[2] - 1),
            dim=-1,
        )
        return torch.where(mask_positions == 1, self.mask_tokens, x), mask_positions

    def _voice_mask(self, x):
        type_index = self.encoding["dimensions"].index("type")
        voice_index = self.encoding["dimensions"].index("instrument")

        voice_count = x[:, :, voice_index].max(dim=1).values + 1
        voices_to_mask = torch.tensor(
            [np.random.randint(0, count) for count in voice_count]
        )

        mask_positions = (
            (
                (torch.rand(x.shape[:2]) < 0.65)
                * (x[:, :, voice_index] == voices_to_mask.unsqueeze(dim=-1))
                * (x[:, :, type_index] == self.encoding["type_code_map"]["note"])
            )
            .unsqueeze(dim=-1)
            .expand_as(x)
        )

        return (
            torch.where(mask_positions == 1, self.mask_tokens, x),
            mask_positions,
        )

    def _mask(self, x, mask_type):
        if mask_type == "random":
            return self._random_mask(x)
        elif mask_type == "voice":
            return self._voice_mask(x)

    def forward(self, x, mask=None):
        mask_indices = None

        if mask is not None:
            x, mask_indices = self._mask(x, mask)

        x = self._embed(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self._reconstruct(x)

        return x, mask_indices
