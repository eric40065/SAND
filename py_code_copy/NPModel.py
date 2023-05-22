import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# class Object(object):
#     pass
# self = Object()

class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_layers, out_dim):
        super().__init__()
        # in_dim, hidden_layers, out_dim = 2, [32, 32], 32
        self.lin0 = nn.Linear(in_dim, hidden_layers[0])
        self.hlin = nn.ModuleList([nn.Linear(hidden_layers[i-1], hidden_layers[i]) for i in range(1, len(hidden_layers))])
        self.out = nn.Linear(hidden_layers[-1], out_dim)

    def forward(self, x):
        # x = IN
        x = F.relu(self.lin0(x))
        for layer in self.hlin:
            x = F.relu(layer(x))
        return self.out(x)


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_layers, out_dim):
        super().__init__()
        self._ff = FeedForward(in_dim, hidden_layers, out_dim)

    def forward(self, x, y, mask):
        # x, y, mask = obs_x, obs_y, obs_mask
        # B, J, _ = x.size()
        assert x.size() == y.size()
        IN = torch.cat((x, y), dim=-1) # (B, J, dim(x) + dim(y) = 2)
        OUT = self._ff(IN).masked_fill(mask == 0, 0)
        return OUT.mean(dim=1) # (B, out_dim)


class Decoder(nn.Module):
    def __init__(self, in_dim, hidden_layers, out_dim=2):
        super().__init__()
        # in_dim, hidden_layers, out_dim = e_out_dim + in_dim - 1, [32, 32], 2
        self._ff = FeedForward(in_dim, hidden_layers, out_dim)
        self._softplus = nn.Softplus()

    def forward(self, e_output, x, mask):
        # e_output, x, mask = representation, trg_x, trg_mask
        B, _ = e_output.size()
        Bx, J, _ = x.size()
        assert B == Bx
        IN = torch.cat((e_output.unsqueeze(1).repeat((1, J, 1)), x), dim=-1) # (B, J, {Enc}out_dim + dim(x))
        OUT = self._ff(IN) # (B, J, 2)
        mu, sd = torch.split(OUT, 1, dim=-1)
        mu = mu.masked_fill(mask == 0, 0)
        sd = sd.masked_fill(mask == 0, 0)
        sigma = 0.1 + 0.9 * self._softplus(sd)
        cov = torch.diag_embed(torch.pow(sigma, 2).squeeze(-1))
        # MultivariateNormal in PyTorch uses cov as its argument
        # MultivariateNormalDiag in TF uses scale, sqrt(cov), as its argument
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu.squeeze(-1), cov)
        return dist, mu, sigma


class CNPModel(nn.Module):
    def __init__(self, in_dim, e_hidden, e_out_dim, d_hidden):
        super().__init__()
        self._encoder = Encoder(in_dim, e_hidden, e_out_dim)
        self._decoder = Decoder(e_out_dim + in_dim - 1, d_hidden)

    def forward(self, obs_x, obs_y, obs_mask, trg_x, trg_y = None, trg_mask = None):
        # obs_x, obs_y, obs_mask, trg_x, trg_y, trg_mask = context_t, context_x, context_mask, trg_t, trg_x, trg_mask
        # obs_x, obs_y, obs_mask, trg_x, trg_mask = context_t, context_x, context_mask, trg_t, trg_mask
        representation = self._encoder(obs_x, obs_y, obs_mask)
        dist, mu, sigma = self._decoder(representation, trg_x, trg_mask)
        if trg_y is not None:
            trg_y = trg_y.masked_fill(trg_mask.squeeze(-1) == 0, 0)
            log_p = dist.log_prob(trg_y)
        else:
            log_p = None

        return log_p, mu, sigma
