import torch
import torch.nn as nn


class SpectralConv3d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 modes_x: int,
                 modes_y: int,
                 modes_t: int):
        """
        Initialize the 3D Spectral Convolution for spatial dimensions x, y and temporal dimension t.
        The spectral convolution is done through a real-valued, parametrized convolution operator acting
        onto a given input. We assume periodic boundary conditions, making this convolution operator periodic.

        We project the convolution into the Fourier space using FFT. In this spectral space, the convolution
        becomes a product between a parametrized weight tensor and the input. We will learn the convolution
        operator in the spectral space, under this weight tensor form, as it allows for efficient implementation
        of the FFT.

        :param in_channels: channels in input of convolution
        :param out_channels: channels in output of convolution
        :param modes_x: Fourier modes to keep in x-dimension
        :param modes_y: Fourier modes to keep in y-dimension
        :param modes_t: Fourier modes to keep in t-dimension
        """

        super(SpectralConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_t = modes_t

        # Scaled weights for each "corner" of the FFT output. See forward pass for more info.
        # Shape: (in_channels, out_channels, modes_x, modes_y, modes_t)
        # Data: complex
        scale = 1.0 / (in_channels * out_channels)
        self.weights_corner1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes_x, self.modes_y, self.modes_t, dtype=torch.cfloat))
        self.weights_corner2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes_x, self.modes_y, self.modes_t, dtype=torch.cfloat))
        self.weights_corner3 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes_x, self.modes_y, self.modes_t, dtype=torch.cfloat))
        self.weights_corner4 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes_x, self.modes_y, self.modes_t, dtype=torch.cfloat))

    def complex_mult3d(self, corner: torch.Tensor, weights: torch.Tensor):
        """
        Tensor Einstein sum between input tensor and weights tensor
        (batch_size, in_channels, x, y, t), (in_channels, out_channels, x, y, t) -> (batch_size, out_channels, x, y, t)
                     corner                               weights                           "corner x weights"

        :param corner: "corner" of the FFT output
        :param weights: weights associated to the "corner"
        :return: (batch_size, out_channels, x, y, t)-shaped tensor, multiplication of the input by the weights
        """

        return torch.einsum("bixyt,ioxyt->boxyt", corner, weights)

    def forward(self, input: torch.Tensor):
        """
        Compute the spectral convolution from given input.

        Since the convolution operator is assumed periodic, it admits a Fourier series expansion, and
        we can work with discrete modes. The idea of using FFT is to truncate high-frenquency modes and
        only keep low-frequency modes of x, y, and t. This severely reduces the number of parameters, thus
        reducing computing time.

        The convolution operator is real-valued, meaning that its resulting FFT (the weight tensor we want
        to learn) has conjugate symmetry. This means that we can use RFFT to reduce the number of value by half.
        It does this symmetry on the last dimension, which will be the t-dimension in our case.

        :param input: (batch_size, in_channels, x, y, t)-shaped tensor
        :return: (batch_size, out_channels, x, y, t)-shaped tensor
        """

        # Compute the FFT of the input for dimensions x, y, t
        input_fft = torch.fft.rfftn(input, dim=(-3, -2, -1))

        # Since we use RFFT, we halve the t-dimension size of the tensor.
        batch_size = input.shape[0]
        output_fft = torch.zeros(
            (batch_size, self.out_channels, input.size(-3), input.size(-2), input.size(-1) // 2 + 1),
            dtype=torch.cfloat, device=input_fft.device)

        # The FFT indexes [-k_max, k_max] to [0, 2k_max - 1]. We want the low frequency modes which are around -k_max
        # and k_max, i.e. in the "corners" [0:modes] and [-modes:]. We only do it for x,y-dimensions because of the
        # conjugate symmetry of the t-dimension.
        output_fft[:, :, :self.modes_x, :self.modes_y, :self.modes_t] = self.complex_mult3d(
            input_fft[:, :, :self.modes_x, :self.modes_y, :self.modes_t], self.weights_corner1)
        output_fft[:, :, -self.modes_x:, :self.modes_y, :self.modes_t] = self.complex_mult3d(
            input_fft[:, :, -self.modes_x:, :self.modes_y, :self.modes_t], self.weights_corner2)
        output_fft[:, :, :self.modes_x, -self.modes_y:, :self.modes_t] = self.complex_mult3d(
            input_fft[:, :, :self.modes_x, -self.modes_y:, :self.modes_t], self.weights_corner3)
        output_fft[:, :, -self.modes_x:, -self.modes_y:, :self.modes_t] = self.complex_mult3d(
            input_fft[:, :, -self.modes_x:, -self.modes_y:, :self.modes_t], self.weights_corner4)

        # Return to the physical space, use IRFFT to accept input from RFFT
        return torch.fft.irfftn(output_fft, s=(input.size(-3), input.size(-2), input.size(-1)))


class FNO3d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int,
                 modes_x: int,
                 modes_y: int,
                 modes_t: int):
        """
        Initialize the model. It contains 4 Fourier layers, and 2 MLPs to lift the input into the higher-dimensional
        Fourier layers and project it back into the output.

        :param in_channels: channels in input of model
        :param out_channels: channels in output of model
        :param mid_channels: channels in input/output of convolution (same)
        :param modes_x: Fourier modes to keep in x-dimension
        :param modes_y: Fourier modes to keep in y-dimension
        :param modes_t: Fourier modes to keep in t-dimension
        """
        super(FNO3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_t = modes_t

        # Lifting of input into higher dimensional representation for spectral convolution
        # Input channel is ??: the solution of the first T timesteps + 3 locations x, y, t
        # (u(1, x, y), ..., u(T, x, y), x, y, t) in the thesis, T = 10
        self.P = nn.Linear(self.in_channels, self.mid_channels)

        # 4 Fourier layers:
        # Fourier integral operator
        self.fourier_layers = nn.ModuleList([
            SpectralConv3d(self.mid_channels, self.mid_channels, self.modes_x, self.modes_y, self.modes_t)
            for _ in range(4)
        ])

        # Bypasses
        self.bypasses = nn.ModuleList([
            nn.Conv3d(self.mid_channels, self.mid_channels, 1) for _ in range(4)
        ])

        # Projecting the Fourier layers' result into output space
        # Output channel is 1: u(x, y)
        self.Q = nn.Linear(self.mid_channels, self.out_channels)

        # Activation function
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model. The steps are:
        1. Lift the input into the higher-dimensional space used in Fourier layers with self.P
        2. Pass through the 4 Fourier layers, with u' = activation((W + K)(u))
        3. Project back into the desired output space with self.Q

        :param x: input (batch_size, x, y, t, mid_channels)-shaped tensor
        :return: output u(x, y)
        """

        # Lift input and permute to have (batch_size, mid_channels, x, y, t)-shaped tensor
        x = self.P(x).permute(0, 4, 1, 2, 3)

        # Apply Fourier layers
        for i in range(4):
            x_fno = self.fourier_layers[i](x)
            x_bypass = self.bypasses[i](x)
            x = self.activation(x_fno + x_bypass)

        # Permute to have (batch_size, x, y, t, mid_channels)-shaped tensor for projection
        x = x.permute(0, 2, 3, 4, 1)

        return self.Q(x)
