import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings,
                self.embedding_dim,
                device=torch.cuda.current_device(),
                dtype=torch.bfloat16,
            )
        )

    def forward(self, input_ids):
        hidden_size = F.embedding(input_ids, self.weight, None, None, 2.0, False, False)
        return hidden_size


class TextEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.embedding_dim = hidden_size

    def forward(self, input_ids):
        return self.word_embeddings(input_ids)


class AudioEmbedding(nn.Module):
    def __init__(
        self,
        audio_vocab_size: int,
        hidden_size: int,
        audio_head_num: int,
        padding_idx: int,
    ):
        super().__init__()
        self.audio_head_num = audio_head_num
        self.padding_idx = padding_idx
        self.audio_embeddings = torch.nn.ModuleList(
            [Embedding(audio_vocab_size, hidden_size) for _ in range(audio_head_num)]
        )

    def forward(self, codecs: torch.Tensor):
        audio_padding_mask = torch.ones_like(codecs[..., 0]).to(codecs.device)
        audio_embeddings = []
        for i in range(self.audio_head_num):
            audio_ids = codecs[:, :, i]
            audio_embedding = self.audio_embeddings[i](audio_ids)
            audio_embedding = (
                audio_embedding.clone()
            )  # clone so that we don't modify a view
            padding_mask = audio_ids == self.padding_idx
            audio_embedding[padding_mask] = 0.0
            audio_padding_mask = torch.logical_and(padding_mask, audio_padding_mask)
            audio_embeddings.append(audio_embedding)
        return audio_embeddings, audio_padding_mask
