# MinAtar_exp

In this experiment, we attempt to apply the proposed experiment setups to more challenging image-based RL tasks. Considered the computation complexity and limited time, we choose the MinAtar environments instead of the Atari environments. We are glad to update the results on the Atari environments in a future version.

In the MinAtar experiments, we ran the following baseline offline RL agents: 
- DQN: a naive offline DQN agent.
- CQL: the discrete version of the CQL algorithm.
- BC: a behavior cloning agent.
- DQN+BC: a combination of DQN and BC which is inspired by the TD3+BC. For each transition pair $(s, a, r, s')$, we use a softmax function to convert $Q(s, a)$ to a categorical distribution, and then we add the negative log likelihood to the TD loss.

```txt
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
```