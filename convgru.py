import torch
from torch import nn


class Conv1dGRUCell(nn.Module):
    def __init__(self, dim_in, in_channels, out_channels,
                 kernel_size, bias):
        """
        :param input_size: int
            Dimension of input tensor.
        :param in_channels: int
            Number of channels of input tensor.
        :param out_channels: int
            Number of channels of output tensor as well as hidden state.
        :param kernel_size: int
            Size of the convolutional kernel.
        :param stride: int
            Size of the convolutional stride.
        :param padding: int
            Size of the convolutional padding.
        :param bias: bool
            Whether or not to include the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(Conv1dGRUCell, self).__init__()
        self.dim_in = dim_in
        self.padding = kernel_size // 2
        self.out_channels = out_channels
        self.bias = bias

        self.i2h_conv = nn.Conv1d(
                    in_channels,
                    out_channels*3,
                    kernel_size,
                    1,
                    self.padding,
                    bias=bias)

        self.h2h_conv = nn.Conv1d(
                    out_channels,
                    out_channels*3,
                    kernel_size,
                    1,
                    self.padding,
                    bias=bias)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.out_channels,
                           self.dim_in, device=device)

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
        rzn_i = self.i2h_conv(x)
        rzn_h = self.h2h_conv(h_prev)

        ri, zi, ni = torch.split(rzn_i, self.out_channels, dim=1)
        rh, zh, nh = torch.split(rzn_h, self.out_channels, dim=1)

        n = torch.tanh(ni + torch.sigmoid(ri + rh) * nh)
        z = torch.sigmoid(zi + zh)
        h_next = (1 - z) * n + z * h_prev
        return h_next


class Conv1dGRU(nn.Module):
    def __init__(self, dim_in, in_channels, out_channels,
                 kernel_size, num_layers, bias):
        """
        :param dim_in: int
            Dimension of input tensor.
        :param in_channels: int
            Number of channels of input tensor.
        :param out_channels: int
            Number of channels of output tensor as well as hidden state.
        :param kernel_size: int
            Size of convolutional kernel.
        :param num_layers: int
            Number of ConvGRU layers.
        :param bias: bool
            Whether or not to add the bias.
        """
        super(Conv1dGRU, self).__init__()

        if isinstance(out_channels, int):
            out_channels_list = [out_channels] * num_layers
        else:
            assert isinstance(out_channels, list) \
                and len(out_channels) == num_layers
            out_channels_list = out_channels
        in_channels_list = [in_channels] + out_channels_list[:-1]
        if not len(in_channels_list) == num_layers:
            raise ValueError("Inconsistent list length.")

        cell_list = [Conv1dGRUCell(dim_in,
                                   cin,
                                   cout,
                                   kernel_size,
                                   bias)
                     for cin, cout in zip(in_channels_list, out_channels_list)]
        self.cell_list = nn.ModuleList(cell_list)
        self.num_layers = num_layers

    def forward(self, input_tensor):
        """
        :param input_tensor: (N, Cin, W, T)
            input tensor.
        :param hidden_state: (N, C_hidden, W, num_layers)
            tensor of hidden state of previous step.
        :return output_tensor: (N, Cout, W, T)
            output tensor.
        """

        hidden_state_list = self._init_hidden(input_tensor.size(1),
                                              input_tensor.device)
        seq_len = input_tensor.size(0)

        input_layer_list = torch.unbind(input_tensor, dim=0)
        for i in range(self.num_layers):
            h = hidden_state_list[i]
            output_layer_list = []
            for t in range(seq_len):
                h = self.cell_list[i](input_layer_list[t], h)
                output_layer_list.append(h)
            input_layer_list = output_layer_list
        output_tensor = torch.stack(output_layer_list, dim=0)

        '''
        h_in_list = hidden_state_list
        output_list = []
        for t in range(seq_len):
            input = input_tensor[t, :, :, :]
            h_out_list = []
            for i in range(self.num_layers):
                output = self.cell_list[i](input, h_in_list[i])
                input = output
                h_out_list.append(output)
            h_in_list = h_out_list
            output_list.append(output)
        output_tensor = torch.stack(output_list, dim=0)
        '''

        return output_tensor

    def _init_hidden(self, batch_size, device):
        init_states_list = [c.init_hidden(batch_size, device)
                            for c in self.cell_list]
        return init_states_list

    def initialize(self):
        for name, param in self.named_parameters():
            print(name, param.shape)
            if 'i2h' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                if 'bias' in name:
                    nn.init.zeros_(param)
            if 'h2h' in name:
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                if 'bias' in name:
                    nn.init.zeros_(param)
