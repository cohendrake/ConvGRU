import torch
from torch import nn, jit
import torch.nn.functional as F
import math


class Conv1dGRUCell(jit.ScriptModule):
    def __init__(self, dim_in, in_channels, out_channels,
                 kernel_size):
        """
        :param dim_in: int
            Dimension of input tensor.
        :param in_channels: int
            Number of channels of input tensor.
        :param out_channels: int
            Number of channels of output tensor as well as hidden state.
        :param kernel_size: int
            Size of the convolutional kernel.
        """
        super(Conv1dGRUCell, self).__init__()
        self.dim_in = dim_in
        self.padding = kernel_size // 2
        self.out_channels = out_channels

        fan_in_i2h = in_channels * kernel_size
        fan_out_i2h = 3 * out_channels * kernel_size
        fan_in_h2h = out_channels * kernel_size
        fan_out_h2h = fan_out_i2h
        a_i2h = math.sqrt(6 / (fan_in_i2h+fan_out_i2h))
        a_h2h = math.sqrt(6 / (fan_in_h2h+fan_out_h2h))

        self.conv1d_i2h_weight = nn.Parameter(
            torch.empty(3*out_channels,
                        in_channels,
                        kernel_size).uniform_(-a_i2h, a_i2h))
        self.conv1d_i2h_bias = nn.Parameter(torch.zeros(3*self.out_channels))
        self.conv1d_h2h_weight = nn.Parameter(
            torch.empty(3*out_channels,
                        out_channels,
                        kernel_size).uniform_(-a_h2h, a_h2h))
        self.conv1d_h2h_bias = nn.Parameter(torch.zeros(3*self.out_channels))

    @jit.script_method
    def forward(self, x, h_prev):
        """
        :param self:
        :param x: (N, C_in, W)
            input tensor
        :param h_prev: (N, C_hidden, W)
            tensor of hidden state of previous step
        :return h_next: (N, C_hidden, W)
            tensor of hidden state of next step
        """
        rzn_i = F.conv1d(input=x,
                         weight=self.conv1d_i2h_weight,
                         bias=self.conv1d_i2h_bias,
                         padding=self.padding)

        rzn_h = F.conv1d(input=h_prev,
                         weight=self.conv1d_h2h_weight,
                         bias=self.conv1d_h2h_bias,
                         padding=self.padding)

        ri, zi, ni = rzn_i.chunk(3, 1)
        rh, zh, nh = rzn_h.chunk(3, 1)

        r = ri + rh
        z = zi + zh
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.sigmoid(ni + r * nh)

        h_next = (1 - z) * n + z * h_prev
        return h_next


class Conv1dGRULayer(jit.ScriptModule):
    def __init__(self, dim_in, in_channels, out_channels,
                 kernel_size):
        """
        :param dim_in: int
            Dimension of input tensor.
        :param in_channels: int
            Number of channels of input tensor.
        :param out_channels: int
            Number of channels of output tensor as well as hidden state.
        :param kernel_size: int
            Size of convolutional kernel.
        """
        super(Conv1dGRULayer, self).__init__()
        self.cell = Conv1dGRUCell(dim_in, in_channels,
                                  out_channels, kernel_size)

    @jit.script_method
    def forward(self, input, h):
        """
        :param input: (T, N, Cin, W)
            input tensor.
        :param hidden_state: (num_layers, N, C_hidden, W)
            tensor of hidden state of previous step.
        :return output: (T, N, Cout, W)
            output tensor.
        """
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            h = self.cell(inputs[i], h)
            outputs += [h]

        return torch.stack(outputs, 0)


class Conv1dGRU(nn.Module):
    def __init__(self, dim_in, in_channels, out_channels,
                 kernel_size, num_layers):
        super(Conv1dGRU, self).__init__()

        if isinstance(out_channels, int):
            cout_list = [out_channels] * num_layers
        else:
            raise NotImplementedError

        cin_list = [in_channels] + cout_list[:-1]
        layer_list = [Conv1dGRULayer(dim_in, cin, cout, kernel_size)
                      for cin, cout in zip(cin_list, cout_list)]
        self.layer_list = nn.ModuleList(layer_list)

    def forward(self, input, h):
        hs = h.unbind(0)
        for i in range(len(hs)):
            output = self.layer_list[i](input, hs[i])
            input = output

        return output
