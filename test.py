import os
import time
import torch
from torch import nn
import numpy as np
from convgru import Conv1dGRULayer, Conv1dGRUCell, Conv1dGRU

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

    input_tensor = torch.tensor(np.random.randn(
        T, batch_size, in_channels, dim_in
    )).float().cuda()

    label_tensor = torch.tensor(np.random.randn(
        T, batch_size, in_channels, dim_in
    )).float().cuda()

    state_tensor = torch.zeros(num_layers, batch_size,
                               out_channels, dim_in).float().cuda()

    device_ids = range(torch.cuda.device_count())
    conv1d_gru = Conv1dGRU(dim_in, in_channels, out_channels, kernel_size,
                           num_layers).float().cuda()
    conv1d_gru = torch.nn.DataParallel(conv1d_gru, device_ids, dim=1)

    opt = torch.optim.Adam(conv1d_gru.parameters(), 1e-3)
    criterion = nn.MSELoss(reduction="mean")

    start = time.time()
    for i in range(10):
        output_tensor = conv1d_gru(input_tensor, state_tensor)   
        loss = criterion(output_tensor, label_tensor)
        print(loss)

        opt.zero_grad()
        loss.backward()
        opt.step()
    end = time.time()
    print(output_tensor.shape)
    print("Cost {:.2f}".format(end-start))

    # print(cell.graph_for(input_tensor, state_tensor))
