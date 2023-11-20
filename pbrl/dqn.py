import torch
import pbrl
import copy
import numpy as np


class DQNAgent:
    def __init__(
        self,
        q_function: torch.nn.Module,
        optimizer,
        replay_buffer,
        gamma: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.qnet = q_function
        self.qnet_target = copy.deepcopy(self.qnet)
        self.observation_size = self.qnet.observation_size
        self.action_size = self.qnet.action_size
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.epsilon = epsilon

    def sync_target(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state):
        state = torch.tensor(state)
        with torch.no_grad():
            if torch.rand(1) < self.epsilon:
                action = torch.randint(0, self.qnet.action_size, (1,))
            else:
                state = state[None, :]
                qs = self.qnet(state)
                action = torch.argmax(qs)
            return action.item()

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
    ):
        state = torch.tensor(state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        next_state = torch.tensor(next_state)
        done = torch.tensor(done)
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return None
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[torch.arange(self.replay_buffer.batch_size), action]
        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1).values
        next_q.detach_()
        target = reward + self.gamma * next_q * (1 - done)
        loss = torch.nn.functional.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
