import torch
import torch.nn as nn


class TinyGRUPredictor(nn.Module):
    """Lightweight GRU predictor used for online LoRA call prediction."""

    def __init__(
        self,
        num_loras: int,
        embed_dim: int = 8,
        hidden_dim: int = 64,
        time_dim: int = 3,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.embedding = nn.Embedding(num_loras, embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim + time_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_loras)

    def forward(self, lora_ids: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(lora_ids)
        if self.time_dim > 0:
            x = torch.cat([emb, time_features], dim=-1)
        else:
            x = emb
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
