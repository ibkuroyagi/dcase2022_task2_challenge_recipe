import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gem(x: torch.Tensor, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"
        )


class PositionalEncoding(torch.nn.Module):
    """Positional encoding module."""

    def __init__(self, d_model, dropout=0.0, maxlen=512):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout)
        self.maxlen = maxlen
        self.xscale = math.sqrt(self.d_model)
        self._initialize_positional_encoding()

    def _initialize_positional_encoding(self):
        pe = torch.zeros(self.maxlen, self.d_model)
        position = torch.arange(0, self.maxlen, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (Tensor): Input tensor (B, T, `*`).
        Returns:
            Tensor: Encoded tensor (B, T, `*`).
        """
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)
