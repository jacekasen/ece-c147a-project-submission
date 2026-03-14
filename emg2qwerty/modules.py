# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
import torchvision.models as tv_models
from torch import nn


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class LSTMEncoder(nn.Module):
    """A stacked bidirectional LSTM encoder.

    Takes input of shape (T, N, num_features) and returns output of shape
    (T, N, hidden_size * 2) when bidirectional=True, or (T, N, hidden_size)
    when bidirectional=False. Unlike TDSConvEncoder, the temporal length T
    is preserved (no shrinkage), so emission_lengths == input_lengths.

    Args:
        num_features (int): Input feature size.
        hidden_size (int): Number of features in the LSTM hidden state.
            (default: 512)
        num_layers (int): Number of stacked LSTM layers. (default: 3)
        dropout (float): Dropout probability applied between LSTM layers.
            Ignored when num_layers == 1. (default: 0.1)
        bidirectional (bool): If True, use a bidirectional LSTM, doubling
            the output feature size. (default: True)
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False,  # expects (T, N, input_size)
        )
        self.output_size = hidden_size * (2 if bidirectional else 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)  # (T, N, hidden_size * num_directions)
        return outputs


class GRUEncoder(nn.Module):
    """A stacked bidirectional GRU encoder.

    Takes input of shape (T, N, num_features) and returns output of shape
    (T, N, hidden_size * 2) when bidirectional=True, or (T, N, hidden_size)
    when bidirectional=False. The temporal length T is preserved (no
    shrinkage), so emission_lengths == input_lengths.

    Args:
        num_features (int): Input feature size.
        hidden_size (int): Number of features in the GRU hidden state.
            (default: 512)
        num_layers (int): Number of stacked GRU layers. (default: 3)
        dropout (float): Dropout probability applied between GRU layers.
            Ignored when num_layers == 1. (default: 0.1)
        bidirectional (bool): If True, use a bidirectional GRU, doubling
            the output feature size. (default: True)
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False,  # expects (T, N, input_size)
        )
        self.output_size = hidden_size * (2 if bidirectional else 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(inputs)  # (T, N, hidden_size * num_directions)
        return outputs


class ResNet18LSTMEncoder(nn.Module):
    """A ResNet18 CNN feature extractor followed by a stacked bidirectional LSTM.

    ResNet18 processes each time step's spectrogram independently to extract
    spatial features, then the LSTM models temporal dependencies across steps.

    The input is the raw normalized spectrogram of shape
    (T, N, num_bands, electrode_channels, freq). ResNet18 is modified to
    accept ``num_bands`` input channels and adapted for small spatial inputs
    (3x3 first conv, no max-pool) to avoid over-downsampling the 16xfreq images.

    Args:
        num_bands (int): Number of input bands (channels to ResNet). (default: 2)
        hidden_size (int): LSTM hidden state size. (default: 512)
        num_lstm_layers (int): Number of stacked LSTM layers. (default: 2)
        dropout (float): Dropout between LSTM layers. (default: 0.1)
        bidirectional (bool): Use bidirectional LSTM. (default: True)
    """

    RESNET_FEATURE_DIM: int = 512  # ResNet18 outputs 512 channels before fc

    def __init__(
        self,
        num_bands: int = 2,
        hidden_size: int = 512,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        resnet = tv_models.resnet18(weights=None)
        # Adapt first conv for num_bands input channels and small spatial size
        # (electrode_channels=16, freq=33): use 3x3 kernel, stride 1, no maxpool
        resnet.conv1 = nn.Conv2d(
            num_bands, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        resnet.maxpool = nn.Identity()

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,  # AdaptiveAvgPool2d(1, 1) -> (N, 512, 1, 1)
        )

        self.lstm = nn.LSTM(
            input_size=self.RESNET_FEATURE_DIM,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=False,  # expects (T, N, input_size)
        )
        self.output_size = hidden_size * (2 if bidirectional else 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_bands, electrode_channels, freq)
        T, N, bands, C, freq = inputs.shape

        # Process all frames in parallel: (T*N, bands, C, freq)
        x = inputs.reshape(T * N, bands, C, freq)
        x = self.backbone(x)          # (T*N, 512, 1, 1)
        x = x.flatten(start_dim=1)    # (T*N, 512)
        x = x.reshape(T, N, -1)       # (T, N, 512)

        outputs, _ = self.lstm(x)     # (T, N, hidden_size * num_directions)
        return outputs


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in 'Attention Is All You Need'.

    Adds position-dependent sine/cosine signals to the input so the
    Transformer can distinguish token order.

    Args:
        d_model (int): Embedding dimension.
        dropout (float): Dropout applied after adding positional encoding.
        max_len (int): Maximum sequence length supported. (default: 5000)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)                    # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)                            # (max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, d_model)
        inputs = inputs + self.pe[: inputs.size(0)]
        return self.dropout(inputs)


class TransformerEncoder(nn.Module):
    """A Transformer encoder with sinusoidal positional encoding.

    Takes input of shape (T, N, num_features) — the output of the
    MultiBandRotationInvariantMLP + Flatten frontend — and returns output
    of the same shape (T, N, d_model). Unlike RNN encoders, the Transformer
    attends to all positions simultaneously in both directions.

    Args:
        num_features (int): Input feature dimension (from the MLP frontend).
        d_model (int): Internal model dimension. If d_model != num_features,
            a linear projection is applied at the input. (default: 256)
        nhead (int): Number of self-attention heads. Must divide d_model.
            (default: 8)
        num_layers (int): Number of TransformerEncoderLayer blocks.
            (default: 4)
        dim_feedforward (int): Hidden size in each layer's FFN. (default: 1024)
        dropout (float): Dropout throughout the encoder. (default: 0.1)
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj = (
            nn.Linear(num_features, d_model) if num_features != d_model else nn.Identity()
        )
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # expects (T, N, d_model)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_size = d_model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        x = self.input_proj(inputs)    # (T, N, d_model)
        x = self.pos_encoding(x)       # (T, N, d_model)
        return self.transformer(x)     # (T, N, d_model)
