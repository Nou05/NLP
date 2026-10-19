import torch
import torch.nn as nn
import math

class SimpleRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(SimpleRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 1. Trọng số từ Input vào Hidden (W_xh)
        self.W_xh = nn.Parameter(torch.Tensor(hidden_size, input_size))
        
        # 2. Trọng số Recurrent từ Hidden trước vào Hidden hiện tại (W_hh)
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        # 3. Bias (b_h)
        self.b_h = nn.Parameter(torch.Tensor(hidden_size))
        
        # Khởi tạo trọng số (Kaiming / Uniform initialization)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x_t: Tensor hình dạng (batch_size, input_size) - vector đầu vào tại timestep t
            h_prev: Tensor hình dạng (batch_size, hidden_size) - hidden state ở timestep (t-1)
        Output:
            h_t: Tensor hình dạng (batch_size, hidden_size) - hidden state mới tại timestep t
        """
        # Công thức: a_t = x_t * W_xh^T + h_prev * W_hh^T + b_h
        # Trong PyTorch (với batch_size ở chiều đầu), dùng F.linear hoặc @ chuyển vị:
        linear_input = torch.matmul(x_t, self.W_xh.T)
        linear_hidden = torch.matmul(h_prev, self.W_hh.T)
        
        a_t = linear_input + linear_hidden + self.b_h
        h_t = torch.tanh(a_t)
        
        return h_t