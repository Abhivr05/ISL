import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    """
    Basic Temporal Convolutional Network block.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.relu2 = nn.ReLU()

        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):

        residual = self.residual(x)

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        # Match sequence length before residual addition
        if x.size(2) != residual.size(2):
            x = x[:, :, :residual.size(2)]

        return x + residual


class TCNModel(nn.Module):
    """
    TCN-based ISL sign classification model.

    Input:
        (batch_size, 30, 225)

    Output:
        (batch_size, num_classes)
    """

    def __init__(
        self,
        input_size=225,
        num_classes=10,
        channels=(64, 128, 128),
        kernel_size=3
    ):
        super().__init__()

        layers = []

        in_channels = input_size

        for i, out_channels in enumerate(channels):

            dilation = 2 ** i

            layers.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation
                )
            )

            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(
            channels[-1],
            num_classes
        )

    def forward(self, x):

        # Input:
        # (batch, sequence, features)
        # (batch, 30, 225)

        # Conv1d expects:
        # (batch, channels, sequence)

        x = x.transpose(1, 2)

        # (batch, 225, 30)
        x = self.tcn(x)

        # Global temporal pooling
        x = self.global_pool(x)

        # (batch, channels, 1)
        x = x.squeeze(-1)

        # Classification
        x = self.classifier(x)

        return x


if __name__ == "__main__":

    # Test the model

    model = TCNModel(
        input_size=225,
        num_classes=10
    )

    dummy_input = torch.randn(4, 30, 225)

    output = model(dummy_input)

    print("===== TCN MODEL TEST =====")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("==========================")