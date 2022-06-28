# Copyright 2022 Ibuki Kuroyanagi

"""Transformer-based encoder modules."""

import torch
import torch.nn as nn
import torchaudio.transforms as T
from asd_tools.models.modules import PositionalEncoding


class Transformer(nn.Module):
    """Transformer-based module."""

    def __init__(
        self,
        n_mels: int,
        num_blocks: int = 4,
        num_heads: int = 4,
        num_feedforward_units: int = 128,
        neck="option-F",
        activation: str = "gelu",
        use_position_encode: bool = False,
        max_position_encode_length: int = 512,
        dropout: float = 0.0,
        out_dim: int = 6,
        embedding_size: int = 128,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 256,
        add_header=False,
    ):
        super().__init__()
        self.use_position_encode = use_position_encode
        self.embedding_size = embedding_size
        self.add_header = add_header
        # Spectrogram extractor
        self.spectrogram_extractor = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            power=1.0,
            n_mels=n_mels,
        )
        # Build encoder.
        if use_position_encode:
            self.position_encode = PositionalEncoding(
                d_model=n_mels,
                dropout=dropout,
                maxlen=max_position_encode_length,
            )
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_mels,
            nhead=num_heads,
            dim_feedforward=num_feedforward_units,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder_transformer = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=num_blocks,
        )
        if neck == "option-D":
            self.neck = nn.Sequential(
                nn.Linear(n_mels, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU(),
                nn.Linear(self.embedding_size, self.embedding_size, bias=True),
            )
        elif neck == "option-F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(n_mels, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU(),
                nn.Linear(self.embedding_size, self.embedding_size, bias=True),
            )
        else:
            self.neck = nn.Sequential(
                nn.Linear(n_mels, self.embedding_size, bias=False),
                nn.BatchNorm1d(self.embedding_size),
                nn.Linear(self.embedding_size, self.embedding_size, bias=False),
            )
        self.machine_head = nn.Linear(1, 1, bias=True)
        self.section_head = nn.Linear(self.embedding_size, out_dim, bias=False)

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (batch_size, wave_length).
        Returns:
            Tensor: Reconstructed inputs (batch_size, sequence_length, n_mels).
            Tensor: Latent variables (batch_size, sequence_length, num_latent_units) if return_latent = True.
        """
        x = self.spectrogram_extractor(x).transpose(2, 1)
        if self.add_header:
            x = torch.cat([x.mean(1)[:, None, :], x], dim=1)
        # (batch_size, sequence_length, n_mels)
        if self.use_position_encode:
            x = self.position_encode(x)
        enc = self.encoder_transformer(x)
        if self.add_header:
            enc = enc[:, 0]
        else:
            enc = enc.mean(1) + enc.max(1)[0]
        embedding = self.neck(enc)
        machine = self.machine_head(
            torch.pow(embedding, 2).sum(dim=1).unsqueeze(1) / self.embedding_size
        )

        section = self.section_head(embedding)
        output_dict = {
            "embedding": embedding,
            "machine": machine,
            "section": section,
        }
        return output_dict
