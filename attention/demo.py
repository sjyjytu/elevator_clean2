import torch
from self_attention_cv import MultiHeadSelfAttention, SelfAttention

# model = MultiHeadSelfAttention(dim=64)
# x = torch.rand(16, 10, 64)  # [batch, tokens, dim]
# mask = torch.zeros(10, 10)  # tokens X tokens
# mask[5:8, 5:8] = 1
# y = model(x, mask)
# print(y)
# print(y.size())

model = SelfAttention(8)
x = torch.rand(1,2,8)
print(x)
y = model(x)
print(y)