# fully_random_ppo.py
import os
import datetime
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal

import gym
from gym import spaces


#  Simple custom Pendulum env
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class CustomPendulumEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, g=10.0):
        super().__init__()
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )

        self.state = None
        self.last_u = None

    def _get_obs(self):
        theta, theta_dot = self.state
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    # NOTE: seed is intentionally ignored for full randomness
    def reset(self, seed=None, **kwargs):
        high = np.array([np.pi, 1.0])
        self.state = np.random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def step(self, u):
        th, thdot = self.state
        g, m, l, dt = self.g, self.m, self.l, self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt
        self.state = np.array([newth, newthdot])

        cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        reward = -float(cost)

        done = False
        return self._get_obs(), reward, done, {}

    def render(self, mode="human"):
        return None

    def close(self):
        pass


#  Hyperparameters
os.makedirs("./results", exist_ok=True)

num_timesteps = 200     # steps per trajectory/episode (T)
num_trajectories = 10   # trajectories per iteration (N)
num_iterations = 250    # PPO iterations (K)
epochs = 100            # optimization epochs per iteration

batch_size = 10
learning_rate = 3e-4
eps = 0.1               # PPO clipping range

gamma = 0.99
lambda_ = 1.0
vf_coef = 1.0
entropy_coef = 0.01


# Utilities
def calc_reward_togo(rewards, gamma=0.99):
    """Return-to-go (discounted sum from each timestep)."""
    n = len(rewards)
    if n == 0:
        return torch.zeros(0, dtype=torch.float32)
    rtg = np.zeros(n, dtype=np.float32)
    rtg[-1] = rewards[-1]
    for i in reversed(range(n - 1)):
        rtg[i] = rewards[i] + gamma * rtg[i + 1]
    return torch.tensor(rtg, dtype=torch.float32)


def calc_advantages(rewards, values, gamma=0.99, lambda_=1.0):
    """GAE(λ) using provided state values."""
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
    values = values.float()
    T = len(rewards)
    adv = torch.zeros(T, dtype=torch.float32)
    gae = 0.0
    values_ext = torch.cat([values, torch.zeros(1, dtype=torch.float32)])
    for t in reversed(range(T)):
        delta = rewards_t[t] + gamma * values_ext[t + 1] - values_ext[t]
        gae = delta + gamma * lambda_ * gae
        adv[t] = gae
    return adv


#  Tiny MLP
class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, activation=nn.functional.relu):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.act = activation

        nn.init.xavier_uniform_(self.layer1.weight, gain=1.0)
        nn.init.constant_(self.layer1.bias, 0.0)
        nn.init.xavier_uniform_(self.layer2.weight, gain=1.0)
        nn.init.constant_(self.layer2.bias, 0.0)
        nn.init.xavier_uniform_(self.layer3.weight, gain=1.0)
        nn.init.constant_(self.layer3.bias, 0.0)

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        return self.layer3(x)


# Simple on-policy buffer
class ReplayMemory:
    def __init__(self, batch_size=10000):
        self.batch_size = batch_size
        self.clear()

    def push(self, state, action, reward, reward_togo, advantage, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.rewards_togo.append(reward_togo)
        self.advantages.append(advantage)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def sample(self):
        states_t = torch.from_numpy(np.array(self.states, dtype=np.float32))
        actions_t = torch.from_numpy(np.array(self.actions, dtype=np.float32))
        rewards_t = torch.tensor(self.rewards, dtype=torch.float32)
        rtg_t = torch.tensor(self.rewards_togo, dtype=torch.float32)
        adv_t = torch.tensor(self.advantages, dtype=torch.float32)
        values_t = torch.tensor(self.values, dtype=torch.float32)
        logp_t = torch.stack(self.log_probs)
        return states_t, actions_t, rewards_t, rtg_t, adv_t, values_t, logp_t, None

    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.rewards_togo, self.advantages, self.values = [], [], []
        self.log_probs = []


# PPO (clipped + advantage)
class PPO:
    def __init__(self):
        self.policy_net = Net(3, 1)
        self.critic_net = Net(3, 1)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy_net.parameters(), "lr": learning_rate},
                {"params": self.critic_net.parameters(), "lr": learning_rate},
            ]
        )

        self.memory = ReplayMemory(batch_size)
        self.cov = torch.diag(torch.full(size=(1,), fill_value=0.5))  # 1D action


        self.env = CustomPendulumEnv()

    def _reset_env(self):
        rs = self.env.reset()  # ignore any seed
        return rs[0] if isinstance(rs, tuple) else rs

    def generate_trajectory(self):
        """Collect one trajectory and store in memory. Returns its total reward."""
        current_state = self._reset_env()

        states, actions, rewards, log_probs = [], [], [], []

        for _ in range(num_timesteps):
            s_t = torch.as_tensor(current_state, dtype=torch.float32)

            mean = 2.0 * torch.tanh(self.policy_net(s_t))  # keep within [-2, 2]
            dist = MultivariateNormal(mean, self.cov)
            sampled_action = dist.sample().detach()
            executed = torch.clamp(sampled_action, -2.0, 2.0)
            logp = dist.log_prob(executed).detach()

            step_out = self.env.step(executed.cpu().numpy())
            if isinstance(step_out, tuple) and len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_out

            states.append(np.asarray(current_state, dtype=np.float32))
            actions.append(np.asarray(executed.cpu().numpy(), dtype=np.float32))
            rewards.append(float(reward))
            log_probs.append(logp)

            current_state = next_state
            if done:
                break

        rtg_t = calc_reward_togo(rewards, gamma)
        states_t = torch.from_numpy(np.array(states, dtype=np.float32))
        with torch.no_grad():
            values_t = self.critic_net(states_t).squeeze(-1)
        adv_t = calc_advantages(rewards, values_t, gamma, lambda_)

        for t in range(len(rewards)):
            self.memory.push(
                states[t],
                actions[t],
                rewards[t],
                float(rtg_t[t].item()),
                float(adv_t[t].item()),
                float(values_t[t].item()),
                log_probs[t],
            )

        return float(np.sum(rewards))

    def train(self):
        scores = []
        scores_window = deque(maxlen=100)

        for it in range(num_iterations):
            # collect N trajectories
            returns_this_iter = [self.generate_trajectory() for _ in range(num_trajectories)]
            mean_ep_return = float(np.mean(returns_this_iter))
            scores.append(mean_ep_return)
            scores_window.append(mean_ep_return)

            # PPO update
            states, actions, rewards, rtg, advantages, values, log_probs, _ = self.memory.sample()
            states = states.float()
            actions = actions.float()
            log_probs = log_probs.float()
            advantages = advantages.float()
            values = values.float()

            for _ in range(epochs):
                mean = 2.0 * torch.tanh(self.policy_net(states))
                dist = MultivariateNormal(mean, self.cov)
                new_log_probs = dist.log_prob(actions)
                ratio = torch.exp(new_log_probs - log_probs)

                adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                new_values = self.critic_net(states).squeeze(-1)
                returns = (advantages + values).detach()

                clipped_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)
                actor_loss = -(torch.min(ratio * adv, clipped_ratio * adv)).mean()
                critic_loss = nn.MSELoss()(new_values, returns)
                entropy = dist.entropy().mean()

                total_loss = actor_loss + vf_coef * critic_loss - entropy_coef * entropy
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            self.memory.clear()

            print(
                "\rIter {:03d}  Total reward: {:.2f}  Average Score: {:.2f}".format(
                    it, mean_ep_return, float(np.mean(scores_window))
                )
            )

        # Save & show reward curve
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        params_str = f"lr_{learning_rate}_eps_{eps}_T_{num_timesteps}_N_{num_trajectories}_K_{num_iterations}_E_{epochs}"
        plt.figure(figsize=(7, 4))
        plt.plot(scores, label="Mean return / iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.title(f"PPO with \n({params_str})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = f"./results/{params_str}_{timestamp}.png"
        plt.savefig(filename, dpi=150)
        plt.show()

    def test(self):
        rs = self.env.reset()
        current_state = rs[0] if isinstance(rs, tuple) else rs
        for _ in range(200):
            s_t = torch.as_tensor(current_state, dtype=torch.float32)
            mean = 2.0 * torch.tanh(self.policy_net(s_t))
            dist = MultivariateNormal(mean, self.cov)
            executed = torch.clamp(dist.sample(), -2.0, 2.0)

            step_out = self.env.step(executed.cpu().numpy())
            if isinstance(step_out, tuple) and len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_out

            current_state = next_state
            if done:
                break
        self.env.close()


# ---------- Entrypoint ----------
if __name__ == "__main__":
    agent = PPO()
    agent.train()
    agent.test()
