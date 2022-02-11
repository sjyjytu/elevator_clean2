import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_heads, dropout, elevator_num, floor_num, device='cpu'):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.elevator_num = elevator_num
        self.floor_num = floor_num

        # d_model // h 仍然是要能整除，换个名字仍然意义不变
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(input_dim, hid_dim)
        self.w_k = nn.Linear(input_dim, hid_dim)
        self.w_v = nn.Linear(input_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, output_dim)
        self.sm = nn.Softmax(dim=1)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        # Q,K,V计算与变形：

        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # Q, K相乘除以scale，这是计算scaled dot product attention的第一步
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # 然后对Q,K相乘的结果计算softmax加上dropout，这是计算scaled dot product attention的第二步：
        attention = self.do(torch.softmax(energy, dim=-1))

        # 第三步，attention结果与V相乘
        x = torch.matmul(attention, V)

        # 最后将多头排列好，就是multi-head attention的结果了
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)

        x = x.view((-1, self.elevator_num, self.floor_num*2))
        x = self.sm(x)

        return x


if __name__ == '__main__':
    A = SelfAttention(64, 64, 64, 4, 0.5)
    x = torch.rand((8, 64))
    y = A(x, x, x)
    print(y.shape)