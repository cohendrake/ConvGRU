import os
import torch
from torch import nn
import numpy as np
from convgru import Conv1dGRU

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    torch.backends.cudnn.benchmark = True
    batch_size = 16
    dim_in = 32
    T = 635
    in_channels = 8
    out_channels = 8
    kernel_size = 7
    num_layers = 2
    bias = True

    # init test
    device_ids = range(torch.cuda.device_count())
    conv1d_gru = Conv1dGRU(dim_in, in_channels, out_channels,
                           kernel_size, num_layers, bias).float()
    conv1d_gru.initialize()
    conv1d_gru = conv1d_gru.cuda()
    conv1d_gru = torch.nn.DataParallel(conv1d_gru, device_ids)

    opt = torch.optim.Adam(conv1d_gru.parameters(), 1e-3)
    criterion = nn.MSELoss(reduction="mean")

    # forward test
    input_tensor = torch.tensor(np.random.randn(
                                    batch_size, in_channels,
                                    dim_in, T).astype(np.float32)).cuda()
    label = torch.tensor(np.random.randn(
                                    batch_size, in_channels,
                                    dim_in, T).astype(np.float32)).cuda()

    for i in range(10):
        output_tensor = conv1d_gru(input_tensor)
        loss = criterion(output_tensor, label)
        print(loss)

        # backward test
        opt.zero_grad()
        loss.backward()
        opt.step()
