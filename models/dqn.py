import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """Tiny CNN network for MinAtar DQN agent."""
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
        # normalize the observations
        # observations = (observations - 0.5) / 0.5
        x = self.net(observations)
        q_values = self.out_layer(x)
        return q_values


class DQNAgent:
    def __init__(self, in_channels, act_dim, args, device, per_alpha=0.6):
        self.act_dim = act_dim
        self.qnet = QNetwork(in_channels, act_dim).to(device)
        self.target_qnet = copy.deepcopy(self.qnet)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=args.lr)
        self.step = 0
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = device
        self.update_step = args.update_step
        self.huber = args.loss == "huber"

        # use per
        self.per = not args.no_per
        self.per_alpha = per_alpha

    def sample_action(self, obs):
        Qs = self.qnet(torch.tensor(obs[None]).to(self.device))
        action = Qs.argmax().item()
        return action

    def loss_fn(self, batch):
        Qs = self.qnet(batch.observations)  # (256, 4, 10, 10)
        Q = torch.gather(Qs, dim=1, index=batch.actions).squeeze()  # (256,)
        with torch.no_grad():
            next_Q = self.target_qnet(batch.next_observations).max(dim=1)[0]  # (256)
        target_Q = batch.rewards + self.gamma * batch.discounts * next_Q
        if self.per:
            if self.huber:
                td_loss = F.smooth_l1_loss(Q, target_Q, reduction="none") * batch.weights
            else:
                td_loss = torch.square(Q - target_Q) * batch.weights
            with torch.no_grad():
                priority = torch.pow(torch.abs(Q - target_Q), self.per_alpha)
        else:
            if self.huber:
                td_loss = F.smooth_l1_loss(Q, target_Q, reduction="none")
            else:
                td_loss = torch.square(Q - target_Q)
        log_info = {
            "avg_Q": Q.mean().item(),
            "max_Q": Q.max().item(),
            "min_Q": Q.min().item(),
            "avg_target_Q": target_Q.mean().item(),
            "max_target_Q": target_Q.max().item(),
            "min_target_Q": target_Q.min().item(),
            "avg_td_loss": td_loss.mean().item(),
            "max_td_loss": td_loss.max().item(),
            "min_td_loss": td_loss.min().item(),
        }
        if self.per:
            log_info.update({"priority": priority,
                             "avg_priority": priority.mean().item(),
                             "max_priority": priority.max().item(),
                             "min_priority": priority.min().item()})
        return td_loss.mean(), log_info

    def update(self, batch):
        self.step += 1
        self.optimizer.zero_grad()
        loss, log_info = self.loss_fn(batch)
        loss.backward()
        self.optimizer.step()

        if self.step % self.update_step == 0:
            for param, target_param in zip(self.qnet.parameters(),
                    self.target_qnet.parameters()):
                target_param.data.copy_(param.data)

        # for param, target_param in zip(self.qnet.parameters(), self.target_qnet.parameters()):
        #     target_param.data.copy_(self.tau*param.data + (1.-self.tau)*target_param)
        return log_info

    def save(self, fname):
        torch.save(self.qnet.state_dict(), fname)

    def load(self, fname):
        self.qnet.load_state_dict(torch.load(fname))