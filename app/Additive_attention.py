import torch.nn as nn
import torch

class AdditiveAttention(nn.Module):
    def __init__(self, head_dim):
        super(AdditiveAttention, self).__init__()
        self.Wa = nn.Linear(head_dim, head_dim)
        self.Ua = nn.Linear(head_dim, head_dim)
        self.V = nn.Linear(head_dim, 1)

    def forward(self, query, keys):
      query = query.unsqueeze(3)
      keys = keys.unsqueeze(2)
      features = torch.tanh(self.Wa(query)+self.Ua(keys))
      scores = self.V(features).squeeze(-1)
      return scores