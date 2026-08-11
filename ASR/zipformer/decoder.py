import torch
import torch.nn as nn
import torch.nn.functional as F
from scaling import Balancer

class Decoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int,
        blank_id: int,
        context_size: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=decoder_dim,
        )
        self.balancer = Balancer(
            decoder_dim,
            channel_dim=-1,
            min_positive=0.0,
            max_positive=1.0,
            min_abs=0.5,
            max_abs=1.0,
            prob=0.05,
        )

        self.blank_id = blank_id

        assert context_size >= 1, context_size
        self.context_size = context_size
        self.vocab_size = vocab_size

        if context_size > 1:
            self.conv = nn.Conv1d(
                in_channels=decoder_dim,
                out_channels=decoder_dim,
                kernel_size=context_size,
                padding=0,
                groups=decoder_dim // 4,  
                bias=False,
            )
            self.balancer2 = Balancer(
                decoder_dim,
                channel_dim=-1,
                min_positive=0.0,
                max_positive=1.0,
                min_abs=0.5,
                max_abs=1.0,
                prob=0.05,
            )
        else:
            self.conv = nn.Identity()
            self.balancer2 = nn.Identity()

    def forward(self, y: torch.Tensor, need_pad: bool = True) -> torch.Tensor:
        y = y.to(torch.int64)
        embedding_out = self.embedding(y.clamp(min=0)) * (y >= 0).unsqueeze(-1)

        embedding_out = self.balancer(embedding_out)

        if self.context_size > 1:
            embedding_out = embedding_out.permute(0, 2, 1)
            if need_pad is True:
                embedding_out = F.pad(embedding_out, pad=(self.context_size - 1, 0))
            else:
                assert embedding_out.size(-1) == self.context_size
            embedding_out = self.conv(embedding_out)
            embedding_out = embedding_out.permute(0, 2, 1)
            embedding_out = F.relu(embedding_out)
            embedding_out = self.balancer2(embedding_out)

        return embedding_out
