import torch
import torch.nn as nn

import numpy as np


class SpectralConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 modes_x: int,
                 modes_y: int):
        """
        Initialize the 2D Spectral Convolution for spatial dimensions x, y. The spectral convolution is done
        through a real-valued, parametrized convolution operator acting onto a given input. We assume periodic
        boundary conditions, making this convolution operator periodic.

        We project the convolution into the Fourier space using FFT. In this spectral space, the convolution
        becomes a product between a parametrized weight tensor and the input. We will learn the convolution
        operator in the spectral space, under this weight tensor form, as it allows for efficient implementation
        of the FFT.

        :param in_channels: channels in input of convolution
        :param out_channels: channels in output of convolution
        :param modes_x: Fourier modes to keep in x-dimension
        :param modes_y: Fourier modes to keep in y-dimension
        """

        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        # Scaled weights for each "corner" of the FFT output. See forward pass for more info.
        # Shape: (in_channels, out_channels, modes_x, modes_y)
        # Data: complex
        scale = 1.0 / (in_channels * out_channels) ** 0.5
        self.weights_corner1 = nn.Parameter(
            scale * torch.rand(self.in_channels, self.out_channels, self.modes_x, self.modes_y, dtype=torch.cfloat))
        self.weights_corner2 = nn.Parameter(
            scale * torch.rand(self.in_channels, self.out_channels, self.modes_x, self.modes_y, dtype=torch.cfloat))

    def complex_mult2d(self, corner: torch.Tensor, weights: torch.Tensor):
        """
        Tensor Einstein sum between input tensor and weights tensor
        (batch_size, in_channels, x, y), (in_channels, out_channels, x, y) -> (batch_size, out_channels, x, y)
                    corner                          weights                         "corner x weights"

        :param corner: "corner" of the FFT output
        :param weights: weights associated to the "corner"
        :return: (batch_size, out_channels, x, y)-shaped tensor, multiplication of the input by the weights
        """

        return torch.einsum("bixy,ioxy->boxy", corner, weights)

    def forward(self, input: torch.Tensor):
        """
        Compute the spectral convolution from given input.

        Since the convolution operator is assumed periodic, it admits a Fourier series expansion, and
        we can work with discrete modes. The idea of using FFT is to truncate high-frenquency modes and
        only keep low-frequency modes of x, and y This severely reduces the number of parameters, thus
        reducing computing time.

        The convolution operator is real-valued, meaning that its resulting FFT (the weight tensor we want
        to learn) has conjugate symmetry. This means that we can use RFFT to reduce the number of value by half.
        It does this symmetry on the last dimension, which will be the y-dimension in our case.

        :param input: (batch_size, in_channels, x, y)-shaped tensor
        :return: (batch_size, out_channels, x, y)-shaped tensor
        """

        # Compute the FFT of the input for dimensions x, y
        input_fft = torch.fft.rfftn(input, dim=(-2, -1))

        # Since we use RFFT, we halve the y-dimension size of the tensor.
        batch_size = input.shape[0]
        output_fft = torch.zeros(
            (batch_size, self.out_channels, input.size(-2), input.size(-1) // 2 + 1),
            dtype=torch.cfloat, device=input_fft.device)
        # The FFT indexes [-k_max, k_max] to [0, 2k_max - 1]. We want the low frequency modes which are around -k_max
        # and k_max, i.e. in the "corners" [0:modes] and [-modes:]. We only do it for the x-axis because of the
        # conjugate symmetry of the y-axis.
        output_fft[:, :, :self.modes_x, :self.modes_y] = self.complex_mult2d(
            input_fft[:, :, :self.modes_x, :self.modes_y], self.weights_corner1)
        output_fft[:, :, -self.modes_x:, :self.modes_y] = self.complex_mult2d(
            input_fft[:, :, -self.modes_x:, :self.modes_y], self.weights_corner2)

        # Return to the physical space, use IRFFT to accept input from RFFT
        return torch.fft.irfftn(output_fft, s=(input.size(-2), input.size(-1)))


class FNO2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels=20,
                 modes_x=12,
                 modes_y=12):
        """
        Initialize the model. It contains 4 Fourier layers, and 2 MLPs to lift the input into the higher-dimensional
        Fourier layers and project it back into the output.

        :param in_channels: channels in input of model
        :param out_channels: channels in output of model
        :param mid_channels: channels in input/output of convolution (same)
        :param modes_x: Fourier modes to keep in x-dimension
        :param modes_y: Fourier modes to keep in y-dimension
        """
        super(FNO2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        # Lifting of input into higher dimensional representation for spectral convolution
        # Input channel is 4: (dcbardx(x, y), dcbardy(x, y), x, y)
        self.P = nn.Linear(self.in_channels, self.mid_channels, bias=False)

        # 4 Fourier layers:
        # Fourier integral operator
        self.fourier_layers = nn.ModuleList([
            SpectralConv2d(self.mid_channels, self.mid_channels, self.modes_x, self.modes_y)
            for _ in range(4)
        ])

        # Bypasses
        self.bypasses = nn.ModuleList([
            nn.Conv2d(self.mid_channels, self.mid_channels, 1) for _ in range(4)
        ])

        # Projecting the Fourier layers' result into output space
        # Output channel is 2: (ucbar(x, y), vcbar(x, y))
        self.Q = nn.Linear(self.mid_channels, self.out_channels, bias=False)

        # Activation function
        self.activation = nn.GELU()

    def mesh_grid(self, shape, device):
        batch_size, size_x, size_y = shape[0], shape[1], shape[2]
        grid_x = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float32)
        grid_x = grid_x.reshape(1, size_x, 1, 1).repeat([batch_size, 1, size_y, 1])
        grid_y = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float32)
        grid_y = grid_y.reshape(1, 1, size_y, 1).repeat([batch_size, size_x, 1, 1])
        return torch.cat((grid_x, grid_y), dim=-1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model. The steps are:
        1. Lift the input into the higher-dimensional space used in Fourier layers with self.P
        2. Pass through the 4 Fourier layers, with u' = activation((W + K)(u))
        3. Project back into the desired output space with self.Q

        :param x: input (batch_size, x, y, in_channels)-shaped tensor
        :return: (ucbar(x, y), vcbar(x, y))
        """

        # Lift input and permute to have (batch_size, mid_channels, x, y)-shaped tensor
        grid = self.mesh_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.P(x)
        x = x.permute(0, 3, 1, 2)

        # Apply Fourier layers
        for i in range(4):
            x_fno = self.fourier_layers[i](x)
            x_bypass = self.bypasses[i](x)
            x = self.activation(x_fno + x_bypass)

        # Permute to have (batch_size, x, y, mid_channels)-shaped tensor for projection
        x = x.permute(0, 2, 3, 1)

        return self.Q(x)
