import copy
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BCNetwork(nn.Module):
    """Tiny CNN network for MinAtar BC agent."""
    def __init__(self, in_channels, act_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=16,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
        )
        self.out_layer = nn.Linear(in_features=128, out_features=act_num)

    def forward(self, observations):
        x = self.net(observations)
        logits = self.out_layer(x)
        distributions = Categorical(logits=logits)
        return distributions


class BCAgent:
    def __init__(self, in_channels, act_dim, args, device):
        self.act_dim = act_dim
        self.bc_net = BCNetwork(in_channels, act_dim).to(device)
        self.optimizer = torch.optim.Adam(self.bc_net.parameters(), lr=args.lr)
        self.update_step = 0
        self.device = device

    def sample_action(self, obs):
        distribution = self.bc_net(torch.tensor(obs[None]).to(self.device))
        action = distribution.sample().item()
        return action

    def loss_fn(self, batch):
        distributions = self.qnet(batch.observations)  # (256, 4, 10, 10)
        log_probs = distributions.log_prob(batch.actions) 
        loss = -log_probs.mean()
        log_info = {
            "avg_logp": log_probs.mean().item(),
            "max_logp": log_probs.max().item(),
            "min_logp": log_probs.min().item(),
        }
        return loss, log_info

    def update(self, batch):
        self.optimizer.zero_grad()
        loss, log_info = self.loss_fn(batch)
        loss.backward()
        self.optimizer.step()
        return log_info

    def save(self, fname):
        torch.save(self.bc_net.state_dict(), fname)

    def load(self, fname):
        self.bc_net.load_state_dict(torch.load(fname))